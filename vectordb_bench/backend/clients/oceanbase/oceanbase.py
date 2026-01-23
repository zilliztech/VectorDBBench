import logging
import struct
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mysql.connector as mysql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import OceanBaseConfigDict, OceanBaseHNSWConfig

log = logging.getLogger(__name__)

OCEANBASE_DEFAULT_LOAD_BATCH_SIZE = 256


class OceanBase(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: OceanBaseConfigDict,
        db_case_config: OceanBaseHNSWConfig,
        collection_name: str = "items",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "OceanBase"
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.table_name = collection_name
        self.load_batch_size = OCEANBASE_DEFAULT_LOAD_BATCH_SIZE
        self._index_name = "vidx"
        self._primary_field = "id"
        self._vector_field = "embedding"

        log.info(
            f"{self.name} initialized with config:\nDatabase: {self.db_config}\nCase Config: {self.db_case_config}"
        )

        self._conn = None
        self._cursor = None

        try:
            self._connect()
            if drop_old:
                self._drop_table()
                self._create_table()
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
            log.exception("Failed to connect to the database")
            raise

    def _disconnect(self):
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        try:
            self._connect()
            self._cursor.execute("SET autocommit=1")

            if self.db_case_config.index in {IndexType.HNSW, IndexType.HNSW_SQ, IndexType.HNSW_BQ}:
                self._cursor.execute(
                    f"SET ob_hnsw_ef_search={(self.db_case_config.search_param())['params']['ef_search']}"
                )
            else:
                self._cursor.execute(
                    f"SET ob_ivf_nprobes={(self.db_case_config.search_param())['params']['ivf_nprobes']}"
                )
            yield
        finally:
            self._disconnect()

    def _drop_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        log.info(f"Dropping table {self.table_name}")
        self._cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    def _create_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        log.info(f"Creating table {self.table_name}")
        create_table_query = f"""
        CREATE TABLE {self.table_name} (
            id INT PRIMARY KEY,
            embedding VECTOR({self.dim})
        );
        """
        self._cursor.execute(create_table_query)

    def optimize(self, data_size: int):
        index_params = self.db_case_config.index_param()
        index_args = ", ".join(f"{k}={v}" for k, v in index_params["params"].items())
        index_query = (
            f"CREATE /*+ PARALLEL(18) */ VECTOR INDEX idx1 "
            f"ON {self.table_name}(embedding) "
            f"WITH (distance={self.db_case_config.parse_metric()}, "
            f"type={index_params['index_type']}, lib={index_params['lib']}, {index_args}"
        )

        if self.db_case_config.index in {IndexType.HNSW, IndexType.HNSW_SQ, IndexType.HNSW_BQ}:
            index_query += ", extra_info_max_size=32"

        index_query += ")"

        log.info("Create index query: %s", index_query)

        try:
            log.info("Creating index...")
            start_time = time.time()
            self._cursor.execute(index_query)
            log.info(f"Index created in {time.time() - start_time:.2f} seconds")

            log.info("Performing major freeze...")
            self._cursor.execute("ALTER SYSTEM MAJOR FREEZE;")
            time.sleep(10)
            self._wait_for_major_compaction()

            log.info("Gathering schema statistics...")
            self._cursor.execute("CALL dbms_stats.gather_schema_stats('test', degree => 96);")
        except mysql.Error:
            log.exception("Failed to optimize index")
            raise

    def need_normalize_cosine(self) -> bool:
        if self.db_case_config.index == IndexType.HNSW_BQ:
            log.info("current HNSW_BQ only supports L2, cosine dataset need normalize.")
            return True

        return False

    def _wait_for_major_compaction(self):
        while True:
            self._cursor.execute(
                "SELECT IF(COUNT(*) = COUNT(STATUS = 'IDLE' OR NULL), 'TRUE', 'FALSE') "
                "AS all_status_idle FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION;"
            )
            all_status_idle = self._cursor.fetchone()[0]
            if all_status_idle == "TRUE":
                break
            time.sleep(10)

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
                self._cursor.execute(
                    f"INSERT /*+ ENABLE_PARALLEL_DML PARALLEL(32) */ INTO {self.table_name} VALUES {values}"  # noqa: S608
                )
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
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"WHERE id == '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Oceanbase - {filters}"
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
            f"SELECT id FROM {self.table_name} "  # noqa: S608
            f"{self.expr} ORDER BY "
            f"{self.db_case_config.parse_metric_func_str()}(embedding, X'{hex_vec}') "
            f"APPROXIMATE LIMIT {k}"
        )

        try:
            self._cursor.execute(query_str)
            return [row[0] for row in self._cursor.fetchall()]
        except mysql.Error:
            log.exception("Failed to execute search query")
            raise
