import logging
import struct
from typing import Dict, Generator, Any, Tuple, Optional
from contextlib import contextmanager
import time

import numpy as np
import mysql.connector as mysql
from ..api import VectorDB, IndexType, MetricType
from .config import OceanBaseIndexConfig, OceanBaseConfigDict

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
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.load_batch_size = OCEANBASE_DEFAULT_LOAD_BATCH_SIZE
        self._index_name = "vidx"
        self._primary_field = "id"
        self._vector_field = "embedding"
        print(self.db_case_config.parse_metric_func_str())
        self._query = f"SELECT /*+ opt_param('rowsets_max_rows', 256)*/ id FROM {self.table_name} ORDER BY {self.db_case_config.parse_metric_func_str()}(embedding, '[%s]') APPROXIMATE LIMIT %s"
        log.info(f"{self.name} config values: {self.db_config}\n{self.db_case_config}\n{self.db_case_config.parse_metric_func_str()}")

        if self.db_config["unix_socket"] != "":
            self._conn = mysql.connect(unix_socket=self.db_config["unix_socket"],
                                       user=self.db_config["user"],
                                       port=self.db_config["port"],
                                       password=self.db_config["password"],
                                       database=self.db_config["database"])
        else:
            self._conn = mysql.connect(host=self.db_config["host"],
                                       user=self.db_config["user"],
                                       port=self.db_config["port"],
                                       password=self.db_config["password"],
                                       database=self.db_config["database"])
        self._cursor = self._conn.cursor()

        if drop_old:
            self._drop_table()
            self._create_table()

        self._cursor.close()
        self._cursor = None
        self._conn.close()
        self._conn = None

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        try:
            if self.db_config["unix_socket"] != "":
                self._conn = mysql.connect(unix_socket=self.db_config["unix_socket"],
                                           user=self.db_config["user"],
                                           port=self.db_config["port"],
                                           password=self.db_config["password"],
                                           database=self.db_config["database"])
            else:
                self._conn = mysql.connect(host=self.db_config["host"],
                                           user=self.db_config["user"],
                                           port=self.db_config["port"],
                                           password=self.db_config["password"],
                                           database=self.db_config["database"])
            self._cursor = self._conn.cursor()
            self._cursor.execute("SET autocommit=1")
            if self.db_case_config.index == IndexType.HNSW or self.db_case_config.index == IndexType.HNSW_SQ:
                self._cursor.execute(f"SET  ob_hnsw_ef_search={(self.db_case_config.search_param())['params']['ef_search']}")
            else:
                self._cursor.execute(f"SET ob_ivf_nprobes={(self.db_case_config.search_param())['params']['ivf_nprobes']}")
            yield
        finally:
            self._cursor.close()
            self._cursor = None
            self._conn.close()
            self._conn = None

    def _drop_table(self):
        if (self._conn is None) or (self._cursor is None):
            raise ValueError("connection is invalid")

        log.info(f"{self.name} client drop table: {self.table_name}")
        self._cursor.execute(
            f"DROP TABLE IF EXISTS {self.table_name}"
        )

    def _create_table(self):
        if (self._conn is None) or (self._cursor is None):
            raise ValueError("connection is invalid")

        log.info(f"{self.name} client create table: {self.table_name}")
        idx_param = self.db_case_config.index_param()
        idx_args_str = ','.join([f"{k}={v}" for k, v in idx_param["params"].items()])
        log.info(
            f"""CREATE TABLE {self.table_name} (
                id INT, 
                embedding vector({self.dim}), 
                primary key(id));"""
        )
        self._cursor.execute(
            f"""CREATE TABLE {self.table_name} (
                id INT, 
                embedding vector({self.dim}), 
                primary key(id));"""
        )

    def ready_to_load(self):
        pass

    def optimize(self, data_size):
        idx_param = self.db_case_config.index_param()
        idx_args_str = ','.join([f"{k}={v}" for k, v in idx_param["params"].items()]) 
        print("begin create index")
        self._cursor.execute(f"create /*+ PARALLEL(32) */ vector index idx1 on items(embedding) with (distance={self.db_case_config.parse_metric()}, type={idx_param['index_type']}, lib={idx_param['lib']}, {idx_args_str})")
        print("begin major freeze") 
        self._cursor.execute("ALTER SYSTEM MAJOR FREEZE;")
        time.sleep(10)
        count = 0
        while count != 1:
            self._cursor.execute("SELECT COUNT(*) FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION WHERE STATUS = 'IDLE';")
            count = self._cursor.fetchone()[0]
            if count != 1:
                time.sleep(10)
        print("major freeze end") 
        self._cursor.execute("call dbms_stats.gather_schema_stats('test',degree=>96);")

    def need_normalize_cosine(self) -> bool:
        return self.db_case_config.metric_type == MetricType.IP or self.db_case_config.metric_type == MetricType.COSINE

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Optional[Exception]]:
        if (self._conn is None) or (self._cursor is None):
            raise ValueError("connection is invalid")

        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.load_batch_size):
                batch_end_offset = min(batch_start_offset + self.load_batch_size, len(embeddings))
                data_batch = [(metadata[i], embeddings[i]) for i in range(batch_start_offset, batch_end_offset)]
                values = ["(%d, '[%s]')" % (i, ",".join([str(e) for e in embedding])) for i, embedding in data_batch]
                values_str = ",".join(values)
                self._cursor.execute(f"insert /*+ ENABLE_PARALLEL_DML PARALLEL(32) */ into {self.table_name} values {values_str}")
                insert_count += (batch_end_offset - batch_start_offset)
        except mysql.Error as e:
            log.info(f"Failed to insert data: {e}")
            return (insert_count, e)
        return (insert_count, None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        if (self._conn is None) or (self._cursor is None):
            raise ValueError("connection is invalid")

        if filters is not None:
            raise ValueError("filters is not supported now")
        self._cursor.execute(self._query % (",".join([str(e) for e in query]), k))
        return [id for id, in self._cursor.fetchall()]
