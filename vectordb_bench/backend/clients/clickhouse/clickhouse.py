"""Wrapper around the Clickhouse vector database over VectorDB"""

import io
import logging
from contextlib import contextmanager
from typing import Any
import clickhouse_connect
import numpy as np

from ..api import VectorDB, DBCaseConfig

log = logging.getLogger(__name__)

class Clickhouse(VectorDB):
    """Use SQLAlchemy instructions"""
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "CHVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "clickhouse_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn =  clickhouse_connect.get_client(
        host=self.db_config["host"],
        port=self.db_config["port"],
        username=self.db_config["user"],
        password=self.db_config["password"],
        database=self.db_config["dbname"])

        if drop_old:
            log.info(f"Clickhouse client drop table : {self.table_name}")
            self._drop_table()
            self._create_table(dim)

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

        self.conn =  clickhouse_connect.get_client(
        host=self.db_config["host"],
        port=self.db_config["port"],
        username=self.db_config["user"],
        password=self.db_config["password"],
        database=self.db_config["dbname"])

        try:
            yield
        finally:
            self.conn.close()
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"

        self.conn.command(f'DROP TABLE IF EXISTS {self.db_config["dbname"]}.{self.table_name}')

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"

        try:
            # create table
            self.conn.command(
                f'CREATE TABLE IF NOT EXISTS {self.db_config["dbname"]}.{self.table_name} \
                    (id UInt32, embedding Array(Float64)) ENGINE = MergeTree() ORDER BY id;'
            )

        except Exception as e:
            log.warning(
                f"Failed to create Clickhouse table: {self.table_name} error: {e}"
            )
            raise e from None

    def ready_to_load(self):
        pass

    def optimize(self, data_size: int | None = None):
        pass

    def ready_to_search(self):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        assert self.conn is not None, "Connection is not initialized"

        try:
            # do not iterate for bulk insert
            items = [metadata, embeddings]

            self.conn.insert(table=self.table_name, data=items,
                             column_names=['id', 'embedding'], column_type_names=['UInt32', 'Array(Float64)'],
                             column_oriented=True)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Clickhouse table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"

        index_param = self.case_config.index_param()
        search_param = self.case_config.search_param()

        if filters:
            gt = filters.get("id")
            filterSql = (f'SELECT id, {search_param["metric_type"]}(embedding,{query}) AS score '
                         f'FROM {self.db_config["dbname"]}.{self.table_name}  '
                         f'WHERE id > {gt} '
                         f'ORDER BY score LIMIT {k};'
                         )
            result = self.conn.query(filterSql).result_rows
            return [int(row[0]) for row in result]
        else:
            selectSql = (f'SELECT id, {search_param["metric_type"]}(embedding,{query}) AS score '
                         f'FROM {self.db_config["dbname"]}.{self.table_name}  '
                         f'ORDER BY score LIMIT {k};'
                         )
            result = self.conn.query(selectSql).result_rows
            return [int(row[0]) for row in result]
