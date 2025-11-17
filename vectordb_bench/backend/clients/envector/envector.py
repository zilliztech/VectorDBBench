"""Wrapper around the EnVector vector database over VectorDB"""

from typing import Any, Dict

import logging
import os
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
# for IVF-FLAT centroid training
# from sklearn.cluster import KMeans
# or for GPU acceleration, we can use 
# from cuml.cluster import KMeans

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
            train_centroids = self.case_config.index_param().get("train_centroids", False)
            
            if index_type == "IVF_FLAT" and train_centroids:
                
                centroid_path = self.case_config.index_param().get("centroids", None)
                is_vct = self.case_config.index_param().get("is_vct", False)
                log.debug(f"IS_VCT: {is_vct}, CENTROIDS: {centroid_path}") # for debug
                
                if centroid_path is not None:
                    if not os.path.exists(centroid_path):
                        raise FileNotFoundError(f"Centroid file {centroid_path} not found for IVF_FLAT index training.")
                    
                    if is_vct:
                        # get VCT trained centroids
                        new_index_params = get_vct_centroids(centroid_path)
                        log.info(f"{self.name} loaded VCT centroids from {centroid_path} for IVF_FLAT index training.")
                        
                        self.vct_params = {
                            "node_batches": new_index_params.pop("node_batches"),
                            "centroid_node_ids": new_index_params.pop("centroid_node_ids"),
                        }

                        self.is_vct = True
                        index_param["virtual_cluster"] = True
                        index_param.update(new_index_params)
                        log.info(f"{self.name} VCT parameters set for IVF_FLAT index creation.")

                    else:
                        # load trained centroids from file
                        centroids = np.load(centroid_path)
                        log.info(f"{self.name} loaded centroids from {centroid_path} for IVF_FLAT index training.")                        

                        # set centroids for index creation
                        index_param["centroids"] = centroids.tolist()

                else:
                    # train centroids using KMeans
                    n_lists = index_param.get("nlist", 250)
                    centroids = get_kmeans_centroids(n_lists)
                    log.info(f"{self.name} No centroid file provided for IVF_FLAT index training, will use random centroids.")

                    # set centroids for index creation
                    index_param["centroids"] = centroids.tolist()

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
            )

        # es2.disconnect()

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
        if self.is_vct:
            return self._insert_vct(embeddings, metadata)
        
        else:
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
    
    def _insert_vct(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
    ) -> tuple[int, Exception]:
        """Insert VCT nodes and their vectors into EnVector."""
        node_batches = self.vct_params.get("node_batches", [])
        insert_count = 0
        try:
            for batch in node_batches:
                node_id = int(batch["node_id"])
                node_vectors = batch.get("vectors")
                if node_vectors is None:
                    continue
                if len(node_vectors) == 0:
                    continue

                vector_count = len(node_vectors)
                log.debug(f"Inserting node {node_id} with {vector_count} vectors") # debug
                
                vector_ids = batch.get("vector_ids") or range(len(node_vectors))
                vectors_list = np.asarray(node_vectors, dtype=np.float32).tolist()
                # metadata = [f"node={node_id},vector={vid}" for vid in vector_ids]
                meta = np.take(metadata, vector_ids).tolist()
                meta = [str(m) for m in meta]

                assert len(vectors_list) == len(meta)

                self.col.insert_vct(vectors_list, metadata=meta, node_id=node_id)

                insert_count += len(vectors_list)

        except Exception as e:
            log.info(f"Failed to insert VCT data: {e}")
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
                res = self._search_vct(query, k)
            
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
            if len(res) > 0 and len(res[0]) > 0:
                return [int(result["metadata"]) for result in res[0] if "metadata" in result]
            else:
                log.warning(f"Unexpected result structure: {res}")
                return []

        except Exception as e:
            log.error(f"Search failed: {e}")
            return []
        
    def _search_vct(
        self,
        query: list[float],
        k: int = 10,
    ):
        """Perform a VCT search on a query embedding and return results."""
        # get params
        centroid_node_ids = self.vct_params.get("centroid_node_ids", [])
        search_params = self.case_config.search_param().get("search_params", {})
        nprobe = search_params.get("nprobe", 6)
        centroids = self.col.index_config.index_params.get("centroids", [])

        # find the nearest centroids
        sims = centroids @ query
        k = max(1, min(nprobe, len(sims)))
        top_indices = np.argpartition(sims, -k)[-k:]
        ordered_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        centroid_list = [int(centroid_node_ids[idx]) for idx in ordered_indices]

        # search
        result = self.col.search_vct(
            query=query,
            top_k=k,
            centroid_list=centroid_list,
            output_fields=["metadata"],
        )

        return result


def get_kmeans_centroids(n_lists: int):
    """Train centroids using KMeans clustering."""
    # kmeans = KMeans(n_clusters=n_lists, n_init=1)
    # kmeans.fit(vectors)
    # centroids = kmeans.cluster_centers_.copy()
    return NotImplementedError("KMeans centroid training cannot be done without dataset.")

def get_vct_centroids(file_path: str) -> Dict[str, Any]:
    """Load VCT centroids from a given file."""
    # load dataset
    centroid_path = "prepared_data.npy"
    tree_path = "preprocessed_data.npy"

    prepared_payload = np.load(os.path.join(file_path, centroid_path), allow_pickle=True).item()

    # centroids
    tree_info = prepared_payload.get("tree")
    nodes_payload = prepared_payload.get("nodes")
    total_nodes = np.uint64(tree_info["total_nodes"])
    nodes = [
        {
            "id": np.uint64(node["id"]),
            "parent": np.uint64(node["parent"]),
        } 
        for node in nodes_payload
    ]
    centroids = prepared_payload.get("centroids")
    clusters_info = prepared_payload.get("clusters")
    centroid_node_ids = [int(cluster["node_id"]) for cluster in clusters_info]

    # tree
    preprocessed_payload = np.load(os.path.join(file_path, tree_path), allow_pickle=True).item()
    
    node_batches = preprocessed_payload.get("nodes")
    node_batches = sorted(node_batches, key=lambda batch: int(batch["node_id"]))

    return {
        "centroids": centroids.tolist(),
        "total_nodes": total_nodes,
        "nodes": nodes,
        "centroid_node_ids": centroid_node_ids,
        "node_batches": node_batches,
    }