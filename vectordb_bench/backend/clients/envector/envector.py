"""Wrapper around the EnVector vector database over VectorDB"""

import logging
import time
import os
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
from sklearn.cluster import KMeans
# for GPU acceleration, we can use cuml.cluster.KMeans

import es2

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import EnVectorIndexConfig


log = logging.getLogger(__name__)


class EnVector(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: EnVectorIndexConfig,
        collection_name: str = "vdbbench",
        drop_old: bool = False,
        name: str = "EnVector",
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the envector vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.batch_size = 128 * 32
        
        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"
        self.col: es2.Index | None = None

        es2.init(
            address=self.db_config.get("uri"), 
            key_path=self.db_config.get("key_path"), 
            key_id=self.db_config.get("key_id"),
            eval_mode=self.case_config.eval_mode,
        )
        if drop_old:
            log.info(f"{self.name} client drop_old index: {self.collection_name}")            
            if self.collection_name in es2.get_index_list():                
                index = es2.Index(self.collection_name)
                index.drop()
        
        # Create the collection
        log.info(f"{self.name} create index: {self.collection_name}")
        # print(f"{self.case_config.index_param().get('params', {})=}")
        if self.collection_name in es2.get_index_list():
            log.info(f"{self.name} index {self.collection_name} already exists, skip creating")
        else:
            index_param = self.case_config.index_param().get("params", {})
            index_type = index_param.get("index_type", "FLAT")
            
            if index_type == "IVF_FLAT" and index_param.get("train_centroids", False):
                
                centroid_path = index_param.get("centroids", None)

                if centroid_path is not None:
                    if not os.path.exists(centroid_path):
                        raise FileNotFoundError(f"Centroid file {centroid_path} not found for IVF_FLAT index training.")
                    centroids = np.load(centroid_path)
                    log.info(f"{self.name} loaded centroids from {centroid_path} for IVF_FLAT index training.")
                else:
                    centroids = None
                    log.info(f"{self.name} No centroid file provided for IVF_FLAT index training, will use random centroids.")
                    
                    # train centroids using KMeans
                    # n_lists = index_param.get("nlist", 250)
                    # kmeans = KMeans(n_clusters=n_lists, n_init=1)
                    # kmeans.fit(vectors)
                    # centroids = kmeans.cluster_centers_.copy()

                index_param["centroids"] = centroids.tolist()
            
            # create index after training centroids
            es2.create_index(
                index_name=self.collection_name,
                dim=dim,
                key_path=self.db_config.get("key_path"),
                key_id=self.db_config.get("key_id"),
                index_params=index_param,
                eval_mode=self.case_config.eval_mode,
            )
        es2.disconnect()

    def __getstate__(self) -> dict:
        """Drop live handles before pickling for multiprocessing."""
        state = self.__dict__.copy()
        state["col"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        es2.init(
            address=self.db_config.get("uri"),
            key_path=self.db_config.get("key_path"),
            key_id=self.db_config.get("key_id"),
            eval_mode=self.case_config.eval_mode,
        )
        try:
            self.col = es2.Index(self.collection_name)
            yield
        finally:
            self.col = None
            es2.disconnect()

    def create_index(self):
        pass

    def _optimize(self):
        pass

    def _post_insert(self):
        pass

    def optimize(self, data_size: int | None = None):
        assert self.col, "Please call self.init() before"
        self._optimize()

    def need_normalize_cosine(self) -> bool:
        """Whether this database need to normalize dataset to support COSINE"""        
        return True

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into EnVector. should call self.init() first"""
        # use the first insert_embeddings to init collection
        assert self.col is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                meta = [str(m) for m in metadata[batch_start_offset:batch_end_offset]]
                vectors = embeddings[batch_start_offset:batch_end_offset]
                self.col.insert(vectors, meta)
                insert_count += len(vectors)
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        pass

    def search_embedding(
        self,
        query: list[float],
        k: int = 10,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.col is not None

        try:
            # Perform the search.
            res = self.col.search(
                query=query,
                top_k=k,
                output_fields=["metadata"],
                search_params=self.case_config.search_param().get("search_params", {}),
            )

            # Handle empty results
            if not res or len(res) == 0:
                log.warning(f"Empty search results for query with k={k}")
                return []

            # Extract metadata from results
            # res structure: [[{id: X, score: Y, metadata: Z}, ...]]
            if len(res) > 0 and len(res[0]) > 0:
                return [int(result["metadata"]) for result in res[0] if "metadata" in result]
            else:
                log.warning(f"Unexpected result structure: {res}")
                return []

        except Exception as e:
            log.error(f"Search failed: {e}")
            return []
