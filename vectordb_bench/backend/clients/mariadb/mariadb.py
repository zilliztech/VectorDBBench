import logging
from contextlib import contextmanager

import mariadb
import numpy as np

from ..api import VectorDB
from .config import MariaDBConfigDict, MariaDBIndexConfig

log = logging.getLogger(__name__)


class MariaDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: MariaDBConfigDict,
        db_case_config: MariaDBIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "MariaDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.db_name = "vectordbbench"
        self.table_name = collection_name
        self.dim = dim

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.db_config)

        if drop_old:
            self._drop_db()
            self._create_db_table(dim)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[mariadb.Connection, mariadb.Cursor]:
        conn = mariadb.connect(**kwargs)
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _drop_db(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop db : {self.db_name}")

        # flush tables before dropping database to avoid some locking issue
        self.cursor.execute("FLUSH TABLES")
        self.cursor.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
        self.cursor.execute("COMMIT")
        self.cursor.execute("FLUSH TABLES")

    def _create_db_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            log.info(f"{self.name} client create database : {self.db_name}")
            self.cursor.execute(f"CREATE DATABASE {self.db_name}")

            log.info(f"{self.name} client create table : {self.table_name}")
            self.cursor.execute(f"USE {self.db_name}")

            self.cursor.execute(
                f"""
              CREATE TABLE {self.table_name} (
                id INT PRIMARY KEY,
                v VECTOR({self.dim}) NOT NULL
              ) ENGINE={index_param["storage_engine"]}
            """
            )
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
        self.conn, self.cursor = self._create_connection(**self.db_config)

        index_param = self.case_config.index_param()
        search_param = self.case_config.search_param()

        # maximize allowed package size
        self.cursor.execute("SET GLOBAL max_allowed_packet = 1073741824")

        if index_param["index_type"] == "HNSW":
            if index_param["max_cache_size"] is not None:
                self.cursor.execute(f"SET GLOBAL mhnsw_max_cache_size = {index_param['max_cache_size']}")
            if search_param["ef_search"] is not None:
                self.cursor.execute(f"SET mhnsw_ef_search = {search_param['ef_search']}")
            self.cursor.execute("COMMIT")

        self.insert_sql = f"INSERT INTO {self.db_name}.{self.table_name} (id, v) VALUES (%s, %s)"  # noqa: S608
        self.select_sql = (
            f"SELECT id FROM {self.db_name}.{self.table_name}"  # noqa: S608
            f"ORDER by vec_distance_{search_param['metric_type']}(v, %s) LIMIT %d"
        )
        self.select_sql_with_filter = (
            f"SELECT id FROM {self.db_name}.{self.table_name} WHERE id >= %d "  # noqa: S608
            f"ORDER by vec_distance_{search_param['metric_type']}(v, %s) LIMIT %d"
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

    def optimize(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            index_options = f"DISTANCE={index_param['metric_type']}"
            if index_param["index_type"] == "HNSW" and index_param["M"] is not None:
                index_options += f" M={index_param['M']}"

            self.cursor.execute(
                f"""
              ALTER TABLE {self.db_name}.{self.table_name}
              ADD VECTOR KEY v(v) {index_options}
            """
            )
            self.cursor.execute("COMMIT")

        except Exception as e:
            log.warning(f"Failed to create index: {self.table_name} error: {e}")
            raise e from None

    @staticmethod
    def vector_to_hex(v):  # noqa: ANN001
        return np.array(v, "float32").tobytes()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            batch_data = []
            for i, row in enumerate(metadata_arr):
                batch_data.append((int(row), self.vector_to_hex(embeddings_arr[i])))

            self.cursor.executemany(self.insert_sql, batch_data)
            self.cursor.execute("COMMIT")
            self.cursor.execute("FLUSH TABLES")

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
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

        search_param = self.case_config.search_param()  # noqa: F841

        if filters:
            self.cursor.execute(self.select_sql_with_filter, (filters.get("id"), self.vector_to_hex(query), k))
        else:
            self.cursor.execute(self.select_sql, (self.vector_to_hex(query), k))

        return [id for (id,) in self.cursor.fetchall()]  # noqa: A001
