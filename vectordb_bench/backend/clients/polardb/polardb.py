import concurrent.futures
import logging
import time
from contextlib import contextmanager

import numpy as np
import pymysql

from ..api import VectorDB
from .config import PolarDBConfigDict, PolarDBIndexConfig

log = logging.getLogger(__name__)


class PolarDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: PolarDBConfigDict,
        db_case_config: PolarDBIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PolarDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        conn, cursor = self._create_connection()

        if drop_old:
            log.info(f"PolarDB dropping old table: {self.table_name}")
            self._create_db_table(cursor, dim)

        cursor.close()
        conn.close()

    def _create_connection(self):
        connect_kwargs = {
            "user": self.db_config["user"],
            "password": self.db_config["password"],
            "autocommit": True,
        }
        if self.db_config.get("unix_socket"):
            connect_kwargs["unix_socket"] = self.db_config["unix_socket"]
        else:
            connect_kwargs["host"] = self.db_config["host"]
            connect_kwargs["port"] = self.db_config["port"]

        conn = pymysql.connect(**connect_kwargs)
        cursor = conn.cursor()
        # Disable query cache to ensure accurate benchmarking
        cursor.execute("SET query_cache_type = OFF")
        return conn, cursor

    def _create_db_table(self, cursor: pymysql.cursors.Cursor, dim: int) -> None:
        index_param = self.case_config.index_param()
        vector_index_comment = index_param["vector_index_comment"]
        post_load_index = getattr(self.case_config, "post_load_index", False)

        try:
            log.info(f"PolarDB creating database: {self.db_config['database']}")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            cursor.execute(f"USE {self.db_config['database']}")
            cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")

            if post_load_index:
                # Post-load mode: create table without vector index, will add via ALTER TABLE later
                create_sql = (
                    f"CREATE TABLE {self.table_name} ("
                    f"id INT PRIMARY KEY, "
                    f"v VECTOR({dim}) NOT NULL"
                    f") ENGINE=InnoDB COMMENT 'COLUMNAR=1'"
                )
                log.info(f"PolarDB creating table (post-load index mode): {create_sql}")
            else:
                # Inline mode: create table with vector index comment
                create_sql = (
                    f"CREATE TABLE {self.table_name} ("
                    f"id INT PRIMARY KEY, "
                    f'v VECTOR({dim}) NOT NULL COMMENT "{vector_index_comment}"'
                    f") ENGINE=InnoDB COMMENT 'COLUMNAR=1'"
                )
                log.info(f"PolarDB creating table: {create_sql}")
            cursor.execute(create_sql)
        except Exception as e:
            log.warning(f"Failed to create table: {self.table_name} error: {e}")
            raise

    @contextmanager
    def init(self):
        self.conn, self.cursor = self._create_connection()

        search_param = self.case_config.search_param()

        # Force PolarDB vector search to use the IMCI engine.
        self.cursor.execute("SET use_imci_engine = FORCED")
        self.cursor.execute("SET imci_enable_vector_search = ON")
        self.cursor.execute("SET imci_max_dop = 1")
        self.cursor.execute("SET cost_threshold_for_imci = 0")

        # Set ef_search
        if search_param.get("ef_search") is not None:
            self.cursor.execute(f"SET polar_vector_index_hnsw_ef_search = {search_param['ef_search']}")

        metric_type = search_param["metric_type"]
        db_name = self.db_config["database"]
        hint = "/*+ SET_VAR(imci_enable_fast_vector_search=on) */"

        self.insert_sql = f"INSERT INTO {db_name}.{self.table_name} (id, v) VALUES (%s, _binary %s)"
        self.select_sql = (
            f"SELECT {hint} id FROM {db_name}.{self.table_name} "
            f"ORDER BY DISTANCE(v, _binary %s, '{metric_type}') "
            f"LIMIT %s"
        )
        self.select_sql_with_filter = (
            f"SELECT id FROM {db_name}.{self.table_name} "
            f"WHERE id >= %s "
            f"ORDER BY DISTANCE(v, _binary %s, '{metric_type}') "
            f"LIMIT %s"
        )

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        db_name = self.db_config["database"]
        index_param = self.case_config.index_param()
        post_load_index = getattr(self.case_config, "post_load_index", False)

        # Ensure vector index builds even for small datasets
        try:
            self.cursor.execute("SET GLOBAL imci_vector_index_dump_rows_threshold = 1")
        except Exception as e:
            log.warning(f"Cannot SET GLOBAL imci_vector_index_dump_rows_threshold (need SUPER): {e}")

        start_time = time.time()

        if post_load_index:
            # Post-load mode: issue ALTER TABLE to add vector index after data load
            vector_index_comment = index_param["vector_index_comment"]
            alter_sql = (
                f"ALTER TABLE {db_name}.{self.table_name} "
                f'MODIFY COLUMN v VECTOR({self.dim}) NOT NULL COMMENT "{vector_index_comment}"'
            )
            log.info(f"PolarDB creating vector index via ALTER TABLE: {alter_sql}")
            self.cursor.execute(alter_sql)
            log.info("ALTER TABLE completed.")

        analyze_sql = f"/* FORCE_IMCI_NODES */ ANALYZE TABLE {db_name}.{self.table_name}"
        log.info(f"PolarDB analyzing table: {analyze_sql}")
        analyze_start = time.time()
        self.cursor.execute(analyze_sql)
        analyze_elapsed = time.time() - analyze_start

        log.info(f"ANALYZE TABLE completed in {analyze_elapsed:.1f}s, waiting for vector index to be built...")

        last_vectors = 0
        while True:
            self.cursor.execute(
                "SELECT VECTORS FROM information_schema.imci_vector_index_stats "
                "WHERE SCHEMA_NAME=%s AND TABLE_NAME=%s",
                (db_name, self.table_name),
            )
            result = self.cursor.fetchone()

            if result is None:
                log.info("Vector index stats not yet available, waiting...")
                time.sleep(2)
                continue

            vectors = int(result[0])

            if vectors != last_vectors:
                elapsed = max(0.0, time.time() - start_time - analyze_elapsed)
                log.info(
                    f"Vector index building: {vectors} vectors indexed "
                    f"(target: {data_size}), elapsed: {elapsed:.1f}s"
                )
                last_vectors = vectors

            if data_size is not None and vectors >= data_size:
                break

            # Also check imci_index_stats for vector_rows as secondary indicator
            self.cursor.execute(
                "SELECT VECTOR_ROWS FROM information_schema.imci_index_stats WHERE SCHEMA_NAME=%s AND TABLE_NAME=%s",
                (db_name, self.table_name),
            )
            idx_result = self.cursor.fetchone()
            if idx_result and data_size is not None and int(idx_result[0]) >= data_size:
                break

            time.sleep(2)

        total_time = max(0.0, time.time() - start_time - analyze_elapsed)
        log.info(f"PolarDB vector index build completed in {total_time:.1f}s")

    @staticmethod
    def vector_to_hex(v: list[float]) -> bytes:
        return np.array(v, "float32").tobytes()

    def _insert_batch(self, embeddings: list[list[float]], metadata: list[int], offset: int, size: int) -> None:
        """Insert a batch of embeddings using a dedicated connection."""
        conn, cursor = self._create_connection()
        try:
            db_name = self.db_config["database"]
            insert_sql = f"INSERT INTO {db_name}.{self.table_name} (id, v) VALUES (%s, _binary %s)"
            batch_data = []
            for i in range(offset, offset + size):
                batch_data.append((int(metadata[i]), self.vector_to_hex(embeddings[i])))

            cursor.executemany(insert_sql, batch_data)
        finally:
            cursor.close()
            conn.close()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            workers = self.case_config.insert_workers
            total = len(embeddings)
            batch_size = max(1, total // workers)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for i in range(0, total, batch_size):
                    offset = i
                    size = min(batch_size, total - i)
                    future = executor.submit(self._insert_batch, embeddings, metadata, offset, size)
                    futures.append(future)

                done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
                for future in done:
                    future.result()
                for future in pending:
                    future.cancel()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            if filters:
                self.cursor.execute(
                    self.select_sql_with_filter,
                    (filters.get("id"), self.vector_to_hex(query), k),
                )
            else:
                self.cursor.execute(self.select_sql, (self.vector_to_hex(query), k))
            return [row[0] for row in self.cursor.fetchall()]
        except Exception:
            log.exception("Failed to execute search query")
            raise
