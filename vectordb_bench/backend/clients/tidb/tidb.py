import concurrent.futures
import io
import logging
import time
from contextlib import contextmanager
from typing import Any

import pymysql

from ..api import VectorDB
from .config import TiDBIndexConfig

log = logging.getLogger(__name__)


class TiDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TiDBIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "TiDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.conn = None  # To be inited by init()
        self.cursor = None  # To be inited by init()

        self.search_fn = db_case_config.search_param()["metric_fn"]

        if drop_old:
            self._drop_table()
            self._create_table()

    @contextmanager
    def init(self):
        with self._get_connection() as (conn, cursor):
            self.conn = conn
            self.cursor = cursor
            try:
                yield
            finally:
                self.conn = None
                self.cursor = None

    @contextmanager
    def _get_connection(self):
        with pymysql.connect(**self.db_config) as conn:
            conn.autocommit = False
            with conn.cursor() as cursor:
                yield conn, cursor

    def _drop_table(self):
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                conn.commit()
        except Exception as e:
            log.warning("Failed to drop table: %s error: %s", self.table_name, e)
            raise

    def _create_table(self):
        try:
            index_param = self.case_config.index_param()
            with self._get_connection() as (conn, cursor):
                cursor.execute(
                    f"""
                    CREATE TABLE {self.table_name} (
                        id BIGINT PRIMARY KEY,
                        embedding VECTOR({self.dim}) NOT NULL,
                        VECTOR INDEX (({index_param["metric_fn"]}(embedding)))
                    );
                    """
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to create table: %s error: %s", self.table_name, e)
            raise

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None) -> None:
        while True:
            progress = self._optimize_check_tiflash_replica_progress()
            if progress != 1:
                log.info("Data replication not ready, progress: %d", progress)
                time.sleep(2)
            else:
                break

        log.info("Waiting TiFlash to catch up...")
        self._optimize_wait_tiflash_catch_up()

        log.info("Start compacting TiFlash replica...")
        self._optimize_compact_tiflash()

        log.info("Waiting index build to finish...")
        log_reduce_seq = 0
        while True:
            pending_rows = self._optimize_get_tiflash_index_pending_rows()
            if pending_rows > 0:
                if log_reduce_seq % 15 == 0:
                    log.info("Index not fully built, pending rows: %d", pending_rows)
                log_reduce_seq += 1
                time.sleep(2)
            else:
                break

        log.info("Index build finished successfully.")

    def _optimize_check_tiflash_replica_progress(self):
        try:
            database = self.db_config["database"]
            with self._get_connection() as (_, cursor):
                cursor.execute(
                    f"""
                    SELECT PROGRESS FROM information_schema.tiflash_replica
                    WHERE TABLE_SCHEMA = "{database}" AND TABLE_NAME = "{self.table_name}"
                    """  # noqa: S608
                )
                result = cursor.fetchone()
                return result[0]
        except Exception as e:
            log.warning("Failed to check TiFlash replica progress: %s", e)
            raise

    def _optimize_wait_tiflash_catch_up(self):
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute('SET @@TIDB_ISOLATION_READ_ENGINES="tidb,tiflash"')
                conn.commit()
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")  # noqa: S608
                result = cursor.fetchone()
                return result[0]
        except Exception as e:
            log.warning("Failed to wait TiFlash to catch up: %s", e)
            raise

    def _optimize_compact_tiflash(self):
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(f"ALTER TABLE {self.table_name} COMPACT")
                conn.commit()
        except Exception as e:
            log.warning("Failed to compact table: %s", e)
            raise

    def _optimize_get_tiflash_index_pending_rows(self):
        try:
            database = self.db_config["database"]
            with self._get_connection() as (_, cursor):
                cursor.execute(
                    f"""
                    SELECT SUM(ROWS_STABLE_NOT_INDEXED)
                    FROM information_schema.tiflash_indexes
                    WHERE TIDB_DATABASE = "{database}" AND TIDB_TABLE = "{self.table_name}"
                    """  # noqa: S608
                )
                result = cursor.fetchone()
                return result[0]
        except Exception as e:
            log.warning("Failed to read TiFlash index pending rows: %s", e)
            raise

    def _insert_embeddings_serial(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        offset: int,
        size: int,
    ) -> Exception:
        try:
            with self._get_connection() as (conn, cursor):
                buf = io.StringIO()
                buf.write(f"INSERT INTO {self.table_name} (id, embedding) VALUES ")  # noqa: S608
                for i in range(offset, offset + size):
                    if i > offset:
                        buf.write(",")
                    buf.write(f'({metadata[i]}, "{embeddings[i]!s}")')
                cursor.execute(buf.getvalue())
                conn.commit()
        except Exception as e:
            log.warning("Failed to insert data into table: %s", e)
            raise

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        workers = 10
        # Avoid exceeding MAX_ALLOWED_PACKET (default=64MB)
        max_batch_size = 64 * 1024 * 1024 // 24 // self.dim
        batch_size = len(embeddings) // workers
        batch_size = min(batch_size, max_batch_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(0, len(embeddings), batch_size):
                offset = i
                size = min(batch_size, len(embeddings) - i)
                future = executor.submit(self._insert_embeddings_serial, embeddings, metadata, offset, size)
                futures.append(future)
            done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
            executor.shutdown(wait=False)
            for future in done:
                future.result()
            for future in pending:
                future.cancel()
        return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        self.cursor.execute(
            f"""
            SELECT id FROM {self.table_name}
            ORDER BY {self.search_fn}(embedding, "{query!s}") LIMIT {k};
            """  # noqa: S608
        )
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]
