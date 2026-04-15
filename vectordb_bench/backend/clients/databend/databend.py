"""Wrapper around the Databend vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Any

from databend_driver import (
    BlockingDatabendClient,
    BlockingDatabendConnection,
)

from .. import IndexType
from ..api import VectorDB
from .config import DatabendConfigDict, DatabendIndexConfig

log = logging.getLogger(__name__)


class Databend(VectorDB):
    """Use SQLAlchemy instructions"""

    def __init__(
        self,
        dim: int,
        db_config: DatabendConfigDict,
        db_case_config: DatabendIndexConfig,
        collection_name: str = "DatabendVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.database_name = db_config['database']
        self.table_name = collection_name
        self.dim = dim

        self.index_param = self.case_config.index_param()
        self.search_param = self.case_config.search_param()
        self.session_param = self.case_config.session_param()

        self._index_name = "databend_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn = self._create_connection(**self.db_config, settings=self.session_param)

        if drop_old:
            log.info(f"Databend client drop table : {self.table_name}")
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.conn.close()
        self.conn = None

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """

        self.conn = self._create_connection(**self.db_config, settings=self.session_param)

        try:
            yield
        finally:
            self.conn.close()
            self.conn = None

    def _create_connection(self, settings: dict | None, **kwargs) -> BlockingDatabendConnection:
        databend_client = BlockingDatabendClient(
            #f"databend://{self.db_config.user}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.db_name}?sslmode=disable"
            f'databend://{self.db_config["user"]}:{self.db_config["password"]}@{self.db_config["host"]}:{self.db_config["port"]}/{self.db_config["database"]}?sslmode=disable'
        )
        return databend_client.get_conn()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        try:
            self.conn.exec(
                f"DROP VECTOR INDEX IF EXISTS {self._index_name} ON {self.database_name}.{self.table_name}"
            )
        except Exception as e:
            log.warning(f"Failed to drop index on table {self.database_name}.{self.table_name}: {e}")
            raise e from None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"

        try:
            self.conn.exec(f"DROP TABLE IF EXISTS {self.database_name}.{self.table_name}")
        except Exception as e:
            log.warning(f"Failed to drop table {self.database_name}.{self.table_name}: {e}")
            raise e from None

    def _perfomance_tuning(self):
        pass

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        try:
            self.conn.exec(
                f"CREATE VECTOR INDEX IF NOT EXISTS {self._index_name} "
                f"ON {self.database_name}.{self.table_name} "
                f'({self._vector_field}) m = {self.index_param["m"]} '
                f'ef_construct = {self.index_param["ef_construct"]} '
                f'distance = {self.index_param["metric_type"]}'
            )
        except Exception as e:
            log.warning(f"Failed to create Databend vector index on table: {self.table_name} error: {e}")
            raise e from None

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"

        try:
            # create table
            self.conn.exec(
                f"CREATE TABLE IF NOT EXISTS {self.database_name}.{self.table_name} "
                f"({self._primary_field} UInt32, "
                f"{self._vector_field} Vector({self.dim})) "
                f"ENGINE = Fuse"
            )

        except Exception as e:
            log.warning(f"Failed to create Databend table: {self.table_name} error: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        assert self.conn is not None, "Connection is not initialized"

        try:
            self.conn.exec(f"OPTIMIZE TABLE {self.database_name}.{self.table_name} ALL")

        except Exception as e:
            log.warning(f"Failed to optimize Databend table: {self.table_name} error: {e}")
            raise e from None

    def _post_insert(self):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        assert self.conn is not None, "Connection is not initialized"

        try:
            rows: List[List[Any]] = []
            for _id, embedding in zip(metadata, embeddings):
                row: List[Any] = [
                    str(_id),
                    str(embedding),
                ]
                rows.append(row)

            self.conn.stream_load(
                f"INSERT INTO {self.database_name}.{self.table_name} VALUES",
                rows,
            )

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Databend table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        if self.case_config.metric_type == "COSINE":
            if filters:
                _id = filters.get("id", 0)
                result = self.conn.query_all(
                    f"SELECT {self._primary_field} "
                    f"FROM {self.database_name}.{self.table_name} "
                    f"WHERE {self._primary_field} > {_id} "
                    f"ORDER BY cosine_distance({self._vector_field}, {query}::Vector({self.dim})) "
                    f"LIMIT {k}",
                )
                return [int(row.values()[0]) for row in result]

            result = self.conn.query_all(
                f"SELECT {self._primary_field} "
                f"FROM {self.database_name}.{self.table_name} "
                f"ORDER BY cosine_distance({self._vector_field}, {query}::Vector({self.dim})) "
                f"LIMIT {k}",
            )
            return [int(row.values()[0]) for row in result]
        if filters:
            _id = filters.get("id", 0)
            result = self.conn.query_all(
                f"SELECT {self._primary_field} "
                f"FROM {self.database_name}.{self.table_name} "
                f"WHERE {self._primary_field} > {_id} "
                f"ORDER BY l2_distance({self._vector_field}, {query}::Vector({self.dim})) "
                f"LIMIT {k}",
            )
            return [int(row.values()[0]) for row in result]

        result = self.conn.query_all(
            f"SELECT {self._primary_field} "
            f"FROM {self.database_name}.{self.table_name} "
            f"ORDER BY l2_distance({self._vector_field}, {query}::Vector({self.dim})) "
            f"LIMIT {k}",
        )
        return [int(row.values()[0]) for row in result]
