import logging
import time
from contextlib import contextmanager

from hyperspace import HyperspaceClient

from ..api import VectorDB
from .config import HyperspaceDBIndexConfig

from vectordb_bench.backend.filter import Filter, FilterOp

log = logging.getLogger(__name__)

class HyperspaceDB(VectorDB):
    supported_filter_types = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: HyperspaceDBIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.expr = None
        
        # Initialize client
        host = db_config.get("host", "localhost:50051")
        api_key = db_config.get("api_key")
        
        client = HyperspaceClient(host, api_key=api_key, pool_size=64)
        
        if drop_old:
            try:
                client.delete_collection(self.collection_name)
                time.sleep(0.5)
            except Exception as e:
                log.info(f"Failed to drop old collection: {e}")
                
        # Create collection
        try:
            metric = self.case_config.parse_metric()
            client.create_collection(
                self.collection_name,
                dimension=dim,
                metric=metric
            )
            # Pre-configure index build parameters
            client.configure(
                ef_search=self.case_config.ef_search,
                ef_construction=self.case_config.ef_construction,
                m=self.case_config.m,
                collection=self.collection_name
            )
        except Exception as e:
            log.warning(f"Failed to create collection: {e}")
            
        client.close()
        
    @contextmanager
    def init(self):
        host = self.db_config.get("host", "localhost:50051")
        api_key = self.db_config.get("api_key")
        self.client = HyperspaceClient(host, api_key=api_key, pool_size=64)
        yield
        self.client.close()
        self.client = None
        
    def ready_to_search(self) -> bool:
        pass

    def need_normalize_cosine(self) -> bool:
        """Whether this database needs to normalize dataset to support COSINE"""
        return True

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            self.expr = [{"type": "range", "key": filters.int_field, "gte": filters.int_value}]
        elif filters.type == FilterOp.StrEqual:
            self.expr = [{"type": "match", "key": filters.label_field, "value": filters.label_value}]
        else:
            raise ValueError(f"Unsupported filter type for HyperspaceDB: {filters.type}")
        
    def optimize(self, data_size: int | None = None):
        # Apply search param config
        try:
            self.client.configure(
                collection=self.collection_name,
                ef_search=self.case_config.ef_search,
                ef_construction=self.case_config.ef_construction,
                m=self.case_config.m,
            )
        except Exception as e:
            log.warning(f"Optimize configure error: {e}")
            
        # Wait for background indexing to complete using gRPC stats
        start_time = time.time()
        timeout = 3600  # 1 hour max
        while True:
            if time.time() - start_time > timeout:
                log.warning("Indexing timeout reached during optimize")
                break
            try:
                stats = self.client.get_collection_stats(self.collection_name)
                queue = stats.get("indexing_queue", 0)
                count = stats.get("count", 0)
                if queue == 0 and count > 0:
                    break
            except Exception as e:
                log.warning(f"Error getting collection stats: {e}")
                time.sleep(5)
                break
            time.sleep(1.0)
            
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        assert self.client is not None, "Please call self.init() before"
        
        # Prepare metadata
        labels_data = kwargs.get("labels_data")
        if labels_data is not None:
            metas = [{"id": str(idx), "labels": str(label)} for idx, label in zip(metadata, labels_data)]
        else:
            metas = [{"id": str(idx)} for idx in metadata]
        
        try:
            self.client.batch_insert(
                vectors=embeddings,
                ids=metadata,
                metadatas=metas,
                collection=self.collection_name
            )
            return len(embeddings), None
        except Exception as e:
            log.warning(f"Failed to insert embeddings: {e}")
            return 0, e
            
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None
    ) -> list[int]:
        assert self.client is not None, "Please call self.init() before"
        
        # Use pre-prepared filter expression if available, fallback to dynamic filters
        expr = self.expr
        if not expr and filters:
            expr = [{"type": "match", "key": k, "value": str(v)} for k, v in filters.items()]
        
        # Direct gRPC fast-path to bypass Python SDK overhead when no filters are present
        if not expr:
            try:
                from hyperspace.proto import hyperspace_pb2
                req = hyperspace_pb2.SearchRequest(
                    vector=query,
                    top_k=k,
                    collection=self.collection_name
                )
                resp = self.client.stub.Search(req, metadata=self.client.metadata)
                return [r.id for r in resp.results]
            except Exception as e:
                log.warning(f"Fast-path search failed: {e}. Falling back to SDK search.")
        
        try:
            results = self.client.search(
                vector=query,
                top_k=k,
                filters=expr,
                collection=self.collection_name
            )
            return [r["id"] for r in results]
        except Exception as e:
            log.warning(f"Failed to search: {e}")
            return []
