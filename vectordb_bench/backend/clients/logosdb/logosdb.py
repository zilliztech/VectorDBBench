import logging
import os
import shutil
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np

from ..api import VectorDB
from .config import LogosDBIndexConfig

log = logging.getLogger(__name__)


class LogosDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: LogosDBIndexConfig,
        collection_name: str = "LogosDBCollection",
        drop_old: bool = False,
        name: str = "LogosDB",
        **kwargs,
    ):
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim
        self.uri = db_config["uri"]
        self.db = None

        if drop_old and os.path.exists(self.uri):
            log.info(f"{self.name} drop_old: removing {self.uri}")
            shutil.rmtree(self.uri)

        import logosdb as _logosdb

        distance = self.case_config.parse_metric()
        db = _logosdb.DB(self.uri, dim=self.dim, distance=distance)
        log.info(f"{self.name} initialized at {self.uri} dim={dim} distance={distance}")
        del db

    @contextmanager
    def init(self):
        import logosdb as _logosdb

        distance = self.case_config.parse_metric()
        self.db = _logosdb.DB(self.uri, dim=self.dim, distance=distance)
        try:
            yield
        finally:
            del self.db
            self.db = None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        assert self.db is not None
        try:
            embeddings_arr = np.array(list(embeddings), dtype=np.float32)
            texts = [str(m) for m in metadata]
            self.db.put_batch(embeddings_arr, texts=texts)
            return len(metadata), None
        except Exception as e:
            log.warning(f"{self.name} insert_embeddings error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.db is not None
        q = np.array(query, dtype=np.float32)
        hits = self.db.search(q, top_k=k)
        return [int(h.text) for h in hits]

    def optimize(self, data_size: int | None = None):
        log.info(f"{self.name} optimize: HNSW index is built incrementally, no explicit step needed")
