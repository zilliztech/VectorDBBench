import logging
from contextlib import contextmanager

import chromadb

from ..api import VectorDB
from .config import ChromaIndexConfig

log = logging.getLogger(__name__)


class ChromaClient(VectorDB):
    """Chroma client for VectorDB.
    To set up Chroma in docker, see https://docs.trychroma.com/usage-guide
    or the instructions in tests/test_chroma.py

    To change to running in process, modify the HttpClient() in __init__() and init().
    """

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ChromaIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        client = chromadb.HttpClient(**db_config)
        assert client.heartbeat() is not None

        if drop_old:
            try:
                client.reset()
            except Exception:
                drop_old = False
                log.info(f"Chroma client drop_old collection: {self.collection_name}")

        self.client = None
        self.collection = None

    @contextmanager
    def init(self):
        self.client = chromadb.HttpClient(**self.db_config)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, configuration=self.case_config.index_param()
        )
        yield
        self.client = None
        self.collection = None

    def ready_to_search(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None):
        assert self.collection is not None, "Please call self.init() before"
        try:
            self.collection.modify(configuration=self.case_config.search_param())
        except Exception as e:
            log.warning(f"Optimize error: {e}")
            raise

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        assert self.collection is not None, "Please call self.init() before"
        ids = [f"{idx}" for idx in metadata]
        metadata = [{"index": mid} for mid in metadata]
        try:
            self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadata)
        except Exception as e:
            log.warning(f"Failed to insert data: {e}")
            return 0, e

        return len(metadata), None

    def search_embedding(
        self, query: list[float], k: int = 100, filters: dict | None = None, timeout: int | None = None
    ) -> list[int]:
        assert self.client is not None, "Please call self.init() before"
        if filters:
            results = self.collection.query(
                query_embeddings=[query], n_results=k, where={"id": {"$gt": filters.get("id")}}
            )
        else:
            results = self.collection.query(query_embeddings=[query], n_results=k)
        return [int(idx) for idx in results["ids"][0]]
