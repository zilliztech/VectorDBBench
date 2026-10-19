import array
import logging
import queue
from contextlib import contextmanager, suppress

import MySQLdb
import numpy as np

from ..api import VectorDB
from .config import AliSQLConfigDict, AliSQLIndexConfig

log = logging.getLogger(__name__)


class AliSQL(VectorDB):
    thread_safe = True

    def __init__(
        self,
        dim: int,
        db_config: AliSQLConfigDict,
        db_case_config: AliSQLIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "AliSQL"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        # Pool of extra connections used to parallelize inserts; built in init().
        self._insert_pool: queue.SimpleQueue | None = None

        self.conn, self.cursor = self._create_connection()

        if drop_old:
            self._drop_db()
            self._create_db_table(dim)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    def _create_connection(self):
        conn = MySQLdb.connect(
            host=self.db_config["host"],
            user=self.db_config["user"],
            port=self.db_config["port"],
            password=self.db_config["password"],
        )
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _acquire_insert_conn(self):
        """Borrow a connection from the insert pool, opening a new one if empty.

        The pool grows lazily to the number of concurrent insert workers: with N
        worker threads at most N connections are checked out at once.
        """
        try:
            return self._insert_pool.get_nowait()
        except queue.Empty:
            conn, cursor = self._create_connection()
            cursor.execute("SET sql_mode = ''")
            return conn, cursor

    def _drain_insert_pool(self):
        """Close every pooled insert connection. Called from init()'s finally, after
        all insert workers have joined, so nothing is checked out at this point.
        """
        pool, self._insert_pool = self._insert_pool, None
        if pool is None:
            return
        while True:
            try:
                conn, cursor = pool.get_nowait()
            except queue.Empty:
                break
            with suppress(Exception):
                cursor.close()
                conn.close()

    def _drop_db(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f'{self.name} client drop db : {self.db_config["database"]}')

        self.cursor.execute(f'DROP DATABASE IF EXISTS {self.db_config["database"]}')
        self.cursor.execute("COMMIT")
        self.cursor.execute("FLUSH TABLES")

    def _create_db_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f'{self.name} client create database : {self.db_config["database"]}')
            self.cursor.execute(f'CREATE DATABASE {self.db_config["database"]}')

            log.info(f"{self.name} client create table : {self.table_name}")
            self.cursor.execute(f'USE {self.db_config["database"]}')

            self.cursor.execute(f"""
              CREATE TABLE {self.table_name} (
                id INT PRIMARY KEY,
                v VECTOR({self.dim}) NOT NULL
              )
            """)
            self.cursor.execute("COMMIT")

        except Exception as e:
            log.warning(f"Failed to create table: {self.table_name} error: {e}")
            raise e from None

    @contextmanager
    def init(self):
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn, self.cursor = self._create_connection()

        index_param = self.case_config.index_param()
        search_param = self.case_config.search_param()

        self.cursor.execute("SET sql_mode = ''")

        if index_param["index_type"] == "HNSW":
            if search_param["ef_search"] is not None:
                self.cursor.execute(f"SET SESSION vidx_hnsw_ef_search = {search_param['ef_search']}")
            self.cursor.execute("COMMIT")

        self.insert_sql = f'INSERT INTO {self.db_config["database"]}.{self.table_name} (id, v) VALUES (%s, %s)'
        self.select_sql = (
            f'SELECT id FROM {self.db_config["database"]}.{self.table_name} '
            f"ORDER by vec_distance_{search_param['metric_type']}(v, %s) LIMIT %s"
        )
        self.select_sql_with_filter = (
            f'SELECT id FROM {self.db_config["database"]}.{self.table_name} WHERE id >= %s '
            f"ORDER by vec_distance_{search_param['metric_type']}(v, %s) LIMIT %s"
        )

        self._search_cursor = self.cursor
        self._insert_pool = queue.SimpleQueue()

        try:
            yield
        finally:
            self._drain_insert_pool()
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None
            self._search_cursor = None

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            index_options = f"DISTANCE={index_param['metric_type']}"
            if index_param["index_type"] == "HNSW" and index_param["M"] is not None:
                index_options += f" M={index_param['M']}"
            if index_param.get("shards") is not None:
                index_options += f" SHARDS={index_param['shards']}"
            if index_param.get("quantization") is not None:
                index_options += f" QUANTIZATION={index_param['quantization']}"

            self.cursor.execute(f"""
              ALTER TABLE {self.db_config["database"]}.{self.table_name}
              ADD VECTOR KEY v(v) {index_options}
            """)
            self.cursor.execute("COMMIT")

        except Exception as e:
            log.warning(f"Failed to create index: {self.table_name} error: {e}")
            raise e from None

    @staticmethod
    def vector_to_hex(v):  # noqa: ANN001
        return array.array("f", v).tobytes()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert one batch of embeddings. Should call self.init() first.

        thread_safe=True: the concurrent insert runner may call this from several
        worker threads sharing this instance. Each call borrows its own connection
        from self._insert_pool, so the workers drive parallel insert streams.
        """
        conn, cursor = self._acquire_insert_conn()
        try:
            embeddings_f32 = np.asarray(embeddings, dtype=np.float32)
            batch_data = [(int(metadata[i]), embeddings_f32[i].tobytes()) for i in range(len(metadata))]

            cursor.executemany(self.insert_sql, batch_data)
            cursor.execute("COMMIT")
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
            # the connection may be left in a bad state; drop it instead of reusing
            with suppress(Exception):
                cursor.close()
                conn.close()
            return 0, e
        else:
            self._insert_pool.put((conn, cursor))
            return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self._search_cursor is not None, "Cursor is not initialized"

        query_bytes = self.vector_to_hex(query)

        try:
            if filters:
                self._search_cursor.execute(self.select_sql_with_filter, (filters.get("id"), query_bytes, k))
            else:
                self._search_cursor.execute(self.select_sql, (query_bytes, k))
            return [row[0] for row in self._search_cursor.fetchall()]

        except MySQLdb.Error:
            log.exception("Failed to execute search query")
            raise
