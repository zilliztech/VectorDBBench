import logging
import re
import struct
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mysql.connector as mysql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import SeekDBConfigDict, SeekDBHNSWConfig

log = logging.getLogger(__name__)

SEEKDB_DEFAULT_LOAD_BATCH_SIZE = 256

# Minimum SeekDB version for dbms_index_manager.refresh() after bulk load (see VERSION()).
_SEEKDB_REFRESH_MIN_VERSION = (1, 3, 0)
_SEEKDB_VERSION_IN_VERSION_STRING = re.compile(r"seekdb-v(\d+(?:\.\d+)*)", re.IGNORECASE)


def _seekdb_version_tuple(version_row: str | None) -> tuple[int, ...] | None:
    if not version_row:
        return None
    m = _SEEKDB_VERSION_IN_VERSION_STRING.search(version_row.strip())
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split(".") if p.isdigit())


def _version_tuple_ge(parsed: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    n = max(len(parsed), len(minimum))
    for i in range(n):
        p = parsed[i] if i < len(parsed) else 0
        m = minimum[i] if i < len(minimum) else 0
        if p != m:
            return p > m
    return True


class SeekDB(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
    ]
    # mysql.connector is not thread-safe; ConcurrentInsertRunner uses max_workers=1 when False.
    # Streaming fixed-rate inserts use rate_runner with per-thread deepcopy+init() instead.
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: SeekDBConfigDict,
        db_case_config: SeekDBHNSWConfig,
        collection_name: str = "items",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "SeekDB"
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.table_name = collection_name
        self.load_batch_size = SEEKDB_DEFAULT_LOAD_BATCH_SIZE
        self._index_name = "vidx"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self.expr = ""

        log.info(
            f"{self.name} initialized with config:\nDatabase: {self.db_config}\nCase Config: {self.db_case_config}"
        )

        self._conn = None
        self._cursor = None

        try:
            self._connect()
            self._apply_system_settings()
            if drop_old:
                self._drop_table()
                self._create_table()
                self._create_index()
        finally:
            self._disconnect()

    def _connect(self):
        try:
            self._conn = mysql.connect(
                host=self.db_config["host"],
                user=self.db_config["user"],
                port=self.db_config["port"],
                password=self.db_config["password"],
                database=self.db_config["database"],
            )
            self._cursor = self._conn.cursor()
        except mysql.Error:
            log.exception("Failed to connect to SeekDB")
            raise

    def _disconnect(self):
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._conn:
            self._conn.close()
            self._conn = None

    def _apply_system_settings(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        self._cursor.execute('ALTER SYSTEM SET memory_limit = "0M"')
        self._cursor.execute("ALTER SYSTEM SET cpu_count = 0")

    def _init_session_settings(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        self._cursor.execute("SET autocommit=1")
        if self.db_case_config.index == IndexType.HNSW:
            ef_search = self.db_case_config.search_param()["params"]["ef_search"]
            # SeekDB uses OceanBase-style session vars (not plain hnsw_ef_search).
            self._cursor.execute(f"SET ob_hnsw_ef_search={ef_search}")

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        try:
            self._connect()
            self._init_session_settings()
            yield
        finally:
            self._disconnect()

    def _drop_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")
        log.info(f"Dropping table {self.table_name}")
        self._cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    def _create_table(self):
        """Create a heap table with a vector column.

        ORGANIZATION HEAP specifies a heap-organized table (no clustered primary
        key order), which is required by SeekDB for vector workloads.
        """
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        log.info(f"Creating heap table {self.table_name}")
        create_table_query = f"""
        CREATE TABLE {self.table_name} (
            id INT,
            embedding VECTOR({self.dim})
        ) ORGANIZATION HEAP;
        """
        self._cursor.execute(create_table_query)

    def _create_index(self):
        """Create the HNSW vector index immediately after table creation.

        Following Milvus's approach: the index is built upfront so that
        streaming inserts are indexed incrementally and searches can run
        concurrently with writes (StreamingPerformanceCase).
        """
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        index_params = self.db_case_config.index_param()
        params = index_params["params"]
        index_args = ", ".join(f"{k}={v}" for k, v in params.items())

        index_query = (
            f"CREATE VECTOR INDEX {self._index_name} "
            f"ON {self.table_name}({self._vector_field}) "
            f"WITH (distance={index_params['metric_type']}, "
            f"type={index_params['index_type']}, {index_args})"
        )

        log.info("Creating HNSW index: %s", index_query)
        try:
            self._cursor.execute(index_query)
            log.info("HNSW index created successfully")
        except mysql.Error:
            log.exception("Failed to create HNSW index")
            raise

    def optimize(self, data_size: int | None = None):
        """Post-load hook: refresh index metadata on SeekDB >= 1.3.0 when available.

        Older releases rely on incremental HNSW indexing only. From 1.3.0 onward,
        ``CALL dbms_index_manager.refresh()`` aligns on-disk index state after bulk
        load (VERSION() strings look like ``... seekdb-v1.3.0.0``).
        """
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        self._cursor.execute("SELECT VERSION()")
        row = self._cursor.fetchone()
        version_str = row[0] if row else None
        parsed = _seekdb_version_tuple(version_str)

        if parsed is None or not _version_tuple_ge(parsed, _SEEKDB_REFRESH_MIN_VERSION):
            log.info(
                "%s optimize: skip dbms_index_manager.refresh (version=%r parsed=%s; need >= %s)",
                self.name,
                version_str,
                parsed,
                ".".join(map(str, _SEEKDB_REFRESH_MIN_VERSION)),
            )
            return

        log.info(
            "%s optimize: SeekDB %s >= %s, calling dbms_index_manager.refresh()",
            self.name,
            ".".join(map(str, parsed)),
            ".".join(map(str, _SEEKDB_REFRESH_MIN_VERSION)),
        )
        try:
            self._cursor.execute("CALL dbms_index_manager.refresh()")
        except mysql.Error:
            log.exception("dbms_index_manager.refresh() failed")
            raise

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        insert_count = 0
        try:
            for batch_start in range(0, len(embeddings), self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, len(embeddings))
                batch = [(metadata[i], embeddings[i]) for i in range(batch_start, batch_end)]
                values = ", ".join(f"({item_id}, '[{','.join(map(str, embedding))}]')" for item_id, embedding in batch)
                self._cursor.execute(f"INSERT INTO {self.table_name} VALUES {values}")
                insert_count += len(batch)
        except mysql.Error:
            log.exception("Failed to insert embeddings")
            raise

        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"WHERE id >= {filters.int_value}"
        else:
            msg = f"Unsupported filter for SeekDB: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        packed = struct.pack(f"<{len(query)}f", *query)
        hex_vec = packed.hex()

        query_str = (
            f"SELECT id FROM {self.table_name} "
            f"{self.expr} ORDER BY "
            f"{self.db_case_config.parse_metric_func_str()}({self._vector_field}, X'{hex_vec}') "
            f"APPROXIMATE LIMIT {k}"
        )

        try:
            self._cursor.execute(query_str)
            return [row[0] for row in self._cursor.fetchall()]
        except mysql.Error:
            log.exception("Failed to execute search query")
            raise
