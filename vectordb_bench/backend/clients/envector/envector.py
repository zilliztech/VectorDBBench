"""Wrapper around the EnVector vector database over VectorDB"""

from typing import Any, Dict

import logging
import os
from collections.abc import Iterable
from contextlib import contextmanager
import pickle

import numpy as np

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

        self.batch_size = 128 * 32 # default batch size for insertions, can be modified for IVF_FLAT
        
        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"
        self.col: es2.Index | None = None

        self.is_vct: bool = False
        self.vct_params: Dict[str, Any] = {}
        kwargs: Dict[str, Any] = {}
        
        es2.init(
            address=self.db_config.get("uri"), 
            key_path=self.db_config.get("key_path"), 
            key_id=self.db_config.get("key_id"),
            eval_mode=self.case_config.eval_mode,
        )
        if drop_old:
            log.info(f"{self.name} client drop_old index: {self.collection_name}")  
            if self.collection_name in es2.get_index_list():          
                es2.drop_index(self.collection_name)
        
        # Create the collection
        log.info(f"{self.name} create index: {self.collection_name}")
        
        if self.collection_name in es2.get_index_list():
            log.info(f"{self.name} index {self.collection_name} already exists, skip creating")
            self.is_vct = self.case_config.index_param().get("is_vct", False)
            log.debug(f"IS_VCT: {self.is_vct}")

        else:
            index_param = self.case_config.index_param().get("params", {})
            index_type = index_param.get("index_type", "FLAT")
            train_centroids = self.case_config.index_param().get("train_centroids", False)
            
            if index_type == "IVF_FLAT" and train_centroids:
                
                centroid_path = self.case_config.index_param().get("centroids_path", None)
                self.is_vct = self.case_config.index_param().get("is_vct", False)
                log.debug(f"IS_VCT: {self.is_vct}")
                
                if centroid_path is not None:
                    if not os.path.exists(centroid_path):
                        raise FileNotFoundError(f"Centroid file {centroid_path} not found for IVF_FLAT index training.")
                    
                    # load trained centroids from file
                    log.debug(f"Centroids: {centroid_path}")
                    centroids = np.load(centroid_path)
                    log.info(f"{self.name} loaded centroids from {centroid_path} for IVF_FLAT index training.")                        

                    # set centroids for index creation
                    index_param["centroids"] = centroids.tolist()

                    if self.is_vct:
                        # set VCT parameters if applicable
                        vct_path = self.case_config.index_param().get("vct_path", None)
                        log.debug(f"VCT: {vct_path}")
                        index_param["virtual_cluster"] = True
                        kwargs["tree_description"] = vct_path
                        self.is_vct = True
                        log.info(f"{self.name} VCT parameters set for IVF_FLAT index creation.")

                else:
                    raise ValueError("Centroids path must be provided for IVF_FLAT index training.")

            # set larger batch size for IVF_FLAT insertions
            if index_type == "IVF_FLAT":
                self.batch_size = int(os.environ.get("NUM_PER_BATCH", 500_000))
                log.debug(
                    f"Set EnVector IVF_FLAT insert batch size to {self.batch_size}. "
                    f"This should be the size of dataset for better performance when IVF_FLAT."
                )

            # create index after training centroids
            es2.create_index(
                index_name=self.collection_name,
                dim=dim,
                key_path=self.db_config.get("key_path"),
                key_id=self.db_config.get("key_id"),
                index_params=index_param,
                eval_mode=self.case_config.eval_mode,
                **kwargs,
            )

        es2.disconnect()

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
            if self.is_vct:
                log.debug(f"VCT: {self.col.index_config.index_param.index_params["virtual_cluster"]}")
                is_vct = self.case_config.index_param().get("is_vct", False)
                assert self.is_vct == is_vct, "is_vct mismatch"
                vct_path = self.case_config.index_param().get("vct_path", None)
                log.debug(f"VCT Path: {vct_path}")
                self.col._load_virtual_cluster_from_pkl(vct_path)
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
        
        log.debug(f"IS_VCT: {self.is_vct}")

        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                meta = [str(m) for m in metadata[batch_start_offset:batch_end_offset]]
                vectors = embeddings[batch_start_offset:batch_end_offset]
                if self.is_vct:
                    self.col.insert_vct(vectors, meta)
                else:
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
            if self.is_vct:
                res = self.col.search_vct(
                    query=query,
                    top_k=k,
                    output_fields=["metadata"],
                    search_params=self.case_config.search_param().get("search_params", {}),
                )
            
            else:
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
            log.debug(f"Search results: {res[0][:3]}")  # Log first 3 results for debugging
            if len(res) > 0 and len(res[0]) > 0:
                return [int(result["metadata"]) for result in res[0] if "metadata" in result]
            else:
                log.warning(f"Unexpected result structure: {res}")
                return []

        except Exception as e:
            log.error(f"Search failed: {e}")
            return []