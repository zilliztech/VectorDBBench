import logging
import time
from contextlib import contextmanager

from hyperspace import HyperspaceClient

from ..api import VectorDB
from .config import HyperspaceDBIndexConfig

log = logging.getLogger(__name__)

class HyperspaceDB(VectorDB):
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
        
    def optimize(self, data_size: int | None = None):
        # Apply search param config
        try:
            self.client.configure(
                collection=self.collection_name,
                ef_search=self.case_config.ef_search,
                ef_construction=self.case_config.ef_construction,
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
        
        # Build filter if present
        flt = None
        if filters:
            # Map simple filter if supported
            pass
            
        try:
            results = self.client.search(
                vector=query,
                top_k=k,
                filter=flt,
                collection=self.collection_name
            )
            return [r["id"] for r in results]
        except Exception as e:
            log.warning(f"Failed to search: {e}")
            return []
