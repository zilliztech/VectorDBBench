import logging
import time
from contextlib import contextmanager
from typing import Generator, Any, Tuple, Optional, List, Dict

import mysql.connector as mysql
import numpy as np
from ..api import VectorDB, IndexType, MetricType
from .config import OceanBaseIndexConfig, OceanBaseConfigDict
import struct

log = logging.getLogger(__name__)

OCEANBASE_DEFAULT_LOAD_BATCH_SIZE = 256


class OceanBase(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: OceanBaseConfigDict,
        db_case_config: OceanBaseIndexConfig,
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
        if self.db_case_config.index == IndexType.HNSW_BQ:
            self.db_case_config.metric_type = MetricType.L2

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
            if self.db_config["unix_socket"]:
                self._conn = mysql.connect(
                    unix_socket=self.db_config["unix_socket"],
                    user=self.db_config["user"],
                    port=self.db_config["port"],
                    password=self.db_config["password"],
                    database=self.db_config["database"],
                )
            else:
                self._conn = mysql.connect(
                    host=self.db_config["host"],
                    user=self.db_config["user"],
                    port=self.db_config["port"],
                    password=self.db_config["password"],
                    database=self.db_config["database"],
                )
            self._cursor = self._conn.cursor()
        except mysql.Error as e:
            log.error(f"Failed to connect to the database: {e}")
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
        create_table_query = (
            f"CREATE TABLE {self.table_name} ("
            f"id INT PRIMARY KEY, "
            f"embedding VECTOR({self.dim})"
            f");"
        )
        self._cursor.execute(create_table_query)

    def optimize(self, data_size: int):
        index_params = self.db_case_config.index_param()
        index_args = ', '.join(f"{k}={v}" for k, v in index_params["params"].items())
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
        except mysql.Error as e:
            log.error(f"Failed to optimize index: {e}")
            raise

    def need_normalize_cosine(self) -> bool:
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
        embeddings: List[List[float]],
        metadata: List[int],
        **kwargs: Any,
    ) -> Tuple[int, Optional[Exception]]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        insert_count = 0
        try:
            for batch_start in range(0, len(embeddings), self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, len(embeddings))
                batch = [
                    (metadata[i], embeddings[i])
                    for i in range(batch_start, batch_end)
                ]
                values = ", ".join(
                    f"({id}, '[{','.join(map(str, embedding))}]')" for id, embedding in batch
                )
                self._cursor.execute(
                    f"INSERT /*+ ENABLE_PARALLEL_DML PARALLEL(32) */ INTO {self.table_name} VALUES {values}"
                )
                insert_count += len(batch)
        except mysql.Error as e:
            log.error(f"Failed to insert embeddings: {e}")
            return insert_count, e

        return insert_count, None

    def search_embedding(
        self,
        query: List[float],
        k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> List[int]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        packed = struct.pack(f'<{len(query)}f', *query)
        hex_vec = packed.hex();
        filter_clause = f"WHERE id >= {filters['id']}" if filters else ""
        query_str = (
            f"SELECT /*+ opt_param('rowsets_max_rows', 256)*/ id FROM {self.table_name} "
            f"{filter_clause} ORDER BY {self.db_case_config.parse_metric_func_str()}(embedding, X'{hex_vec}') APPROXIMATE LIMIT {k}"
        )

        try:
            self._cursor.execute(query_str)
            return [row[0] for row in self._cursor.fetchall()]
        except mysql.Error as e:
            log.error(f"Failed to execute search query: {e}")
            raise