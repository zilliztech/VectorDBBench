"""KiviDB client: a multi-threaded, Redis-compatible store with RediSearch-style
`FT.*` vector search built in (no module to load).

Speaks RESP2 with raw `FT.*` commands so reply parsing does not depend on the
redis-py search helpers or on the negotiated protocol version.
"""

import logging
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import redis

from ...filter import Filter, FilterOp
from ..api import VectorDB
from .config import KiviDBIndexConfig

log = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "vdbbench_kividb"
ID_FIELD = "id"
LABEL_FIELD = "labels"
VECTOR_FIELD = "vector"
PIPELINE_BATCH_SIZE = 1000
INDEX_WAIT_TIMEOUT_S = 3600

# RediSearch TAG syntax: anything but letters, digits and `_` must be escaped.
_TAG_ESCAPE = re.compile(r"([^A-Za-z0-9_])")


def escape_tag_value(value: str) -> str:
    return _TAG_ESCAPE.sub(r"\\\1", value)


class KiviDB(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]
    name = "KiviDB"

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: KiviDBIndexConfig,
        collection_name: str = DEFAULT_INDEX_NAME,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_name = collection_name or DEFAULT_INDEX_NAME
        self.key_prefix = f"{self.index_name}:"
        self.with_scalar_labels = with_scalar_labels
        self.filter_expr = "*"
        self.conn: redis.Redis | None = None

        conn = self._connect()
        info = conn.info("server")
        log.info(
            f"Connected to KiviDB {info.get('kividb_version', 'unknown')} "
            f"(redis_version {info.get('redis_version', 'unknown')})"
        )
        if drop_old:
            self._drop(conn)
        self._create_index(dim, conn)
        conn.close()

    def _connect(self) -> redis.Redis:
        return redis.Redis(
            host=self.db_config["host"],
            port=self.db_config["port"],
            password=self.db_config["password"],
            ssl=self.db_config.get("ssl", False),
            protocol=2,
        )

    def _index_exists(self, conn: redis.Redis) -> bool:
        try:
            conn.execute_command("FT.INFO", self.index_name)
        except redis.exceptions.ResponseError:
            return False
        return True

    def _drop(self, conn: redis.Redis):
        if self._index_exists(conn):
            conn.execute_command("FT.DROPINDEX", self.index_name)
        deleted = 0
        batch: list[bytes] = []
        for key in conn.scan_iter(match=f"{self.key_prefix}*", count=10_000):
            batch.append(key)
            if len(batch) >= 10_000:
                deleted += conn.unlink(*batch)
                batch = []
        if batch:
            deleted += conn.unlink(*batch)
        log.info(f"Dropped KiviDB index {self.index_name} and {deleted} keys")

    def _create_index(self, dim: int, conn: redis.Redis):
        if self._index_exists(conn):
            return
        index_param = self.case_config.index_param()
        vector_attrs = ["TYPE", "FLOAT32", "DIM", dim, "DISTANCE_METRIC", index_param["metric"]]
        if index_param["index_type"] == "HNSW":
            vector_attrs += ["M", index_param["m"], "EF_CONSTRUCTION", index_param["ef_construction"]]
        schema = [ID_FIELD, "NUMERIC"]
        if self.with_scalar_labels:
            schema += [LABEL_FIELD, "TAG"]
        schema += [VECTOR_FIELD, "VECTOR", index_param["index_type"], len(vector_attrs), *vector_attrs]
        conn.execute_command(
            "FT.CREATE", self.index_name, "ON", "HASH", "PREFIX", "1", self.key_prefix, "SCHEMA", *schema
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn = self._connect()
        ef_runtime = self.case_config.search_param()["ef_runtime"]
        self.ef_clause = f" EF_RUNTIME {ef_runtime}" if ef_runtime else ""
        yield
        self.conn.close()
        self.conn = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "call init() first"
        if self.with_scalar_labels and labels_data is None:
            return 0, ValueError("labels_data is required when with_scalar_labels is set")
        try:
            with self.conn.pipeline(transaction=False) as pipe:
                for i, embedding in enumerate(embeddings):
                    mapping = {
                        ID_FIELD: metadata[i],
                        VECTOR_FIELD: np.asarray(embedding, dtype=np.float32).tobytes(),
                    }
                    if self.with_scalar_labels:
                        mapping[LABEL_FIELD] = labels_data[i]
                    pipe.hset(f"{self.key_prefix}{metadata[i]}", mapping=mapping)
                    if (i + 1) % PIPELINE_BATCH_SIZE == 0:
                        pipe.execute()
                pipe.execute()
        except Exception as e:
            log.warning(f"KiviDB insert failed: {e}")
            return 0, e
        return len(embeddings), None

    def _num_docs(self, conn: redis.Redis) -> int:
        info = conn.execute_command("FT.INFO", self.index_name)
        fields = dict(zip(info[0::2], info[1::2], strict=False))
        return int(fields.get(b"num_docs", 0))

    def optimize(self, data_size: int | None = None):
        """KiviDB indexes each vector inside the HSET that stores it; this only
        confirms the index holds the whole corpus before search begins."""
        if not data_size:
            return
        conn = self._connect()
        deadline = time.monotonic() + INDEX_WAIT_TIMEOUT_S
        while (indexed := self._num_docs(conn)) < data_size:
            if time.monotonic() > deadline:
                conn.close()
                msg = f"KiviDB index {self.index_name} has {indexed} of {data_size} docs after {INDEX_WAIT_TIMEOUT_S}s"
                raise TimeoutError(msg)
            time.sleep(1)
        conn.close()

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.filter_expr = "*"
        elif filters.type == FilterOp.NumGE:
            self.filter_expr = f"(@{ID_FIELD}:[{int(filters.int_value)} +inf])"
        elif filters.type == FilterOp.StrEqual:
            self.filter_expr = f"(@{LABEL_FIELD}:{{{escape_tag_value(filters.label_value)}}})"
        else:
            msg = f"Unsupported filter for KiviDB: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "call init() first"
        res = self.conn.execute_command(
            "FT.SEARCH",
            self.index_name,
            f"{self.filter_expr}=>[KNN {k} @{VECTOR_FIELD} $vec{self.ef_clause} AS score]",
            "PARAMS",
            "2",
            "vec",
            np.asarray(query, dtype=np.float32).tobytes(),
            "SORTBY",
            "score",
            "NOCONTENT",
            "LIMIT",
            "0",
            str(k),
            "DIALECT",
            "2",
        )
        # RESP2 reply with NOCONTENT: [total, key1, key2, ...]
        return [int(key.rsplit(b":", 1)[1]) for key in res[1:]]
