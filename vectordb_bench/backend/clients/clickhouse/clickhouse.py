"""Wrapper around the Clickhouse vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver import Client

from .. import IndexType
from ..api import VectorDB
from .config import ClickhouseConfigDict, ClickhouseIndexConfig

log = logging.getLogger(__name__)


class Clickhouse(VectorDB):
    """Use SQLAlchemy instructions"""

    def __init__(
        self,
        dim: int,
        db_config: ClickhouseConfigDict,
        db_case_config: ClickhouseIndexConfig,
        collection_name: str = "CHVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self.index_param = self.case_config.index_param()
        self.search_param = self.case_config.search_param()
        self.session_param = self.case_config.session_param()

        self._index_name = "clickhouse_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn = self._create_connection(**self.db_config, settings=self.session_param)

        if drop_old:
            log.info(f"Clickhouse client drop table : {self.table_name}")
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

    def _create_connection(self, settings: dict | None, **kwargs) -> Client:
        return clickhouse_connect.get_client(**self.db_config, settings=settings)

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        try:
            self.conn.command(
                f'ALTER TABLE {self.db_config["database"]}.{self.table_name} DROP INDEX {self._index_name}'
            )
        except Exception as e:
            log.warning(f"Failed to drop index on table {self.db_config['database']}.{self.table_name}: {e}")
            raise e from None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"

        try:
            self.conn.command(f'DROP TABLE IF EXISTS {self.db_config["database"]}.{self.table_name}')
        except Exception as e:
            log.warning(f"Failed to drop table {self.db_config['database']}.{self.table_name}: {e}")
            raise e from None

    def _perfomance_tuning(self):
        self.conn.command("SET materialize_skip_indexes_on_insert = 1")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        try:
            if self.index_param["index_type"] == IndexType.HNSW.value:
                if (
                    self.index_param["quantization"]
                    and self.index_param["params"]["M"]
                    and self.index_param["params"]["efConstruction"]
                ):
                    query = f"""
                        ALTER TABLE {self.db_config["database"]}.{self.table_name}
                        ADD INDEX {self._index_name} {self._vector_field}
                        TYPE vector_similarity('hnsw', '{self.index_param["metric_type"]}',
                        '{self.index_param["quantization"]}',
                        {self.index_param["params"]["M"]}, {self.index_param["params"]["efConstruction"]})
                        GRANULARITY {self.index_param["granularity"]}
                        """
                else:
                    query = f"""
                        ALTER TABLE {self.db_config["database"]}.{self.table_name}
                        ADD INDEX {self._index_name} {self._vector_field}
                        TYPE vector_similarity('hnsw', '{self.index_param["metric_type"]}')
                        GRANULARITY {self.index_param["granularity"]}
                        """
                self.conn.command(cmd=query)
            else:
                log.warning("HNSW is only avaliable method in clickhouse now")
        except Exception as e:
            log.warning(f"Failed to create Clickhouse vector index on table: {self.table_name} error: {e}")
            raise e from None

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"

        try:
            # create table
            self.conn.command(
                f'CREATE TABLE IF NOT EXISTS {self.db_config["database"]}.{self.table_name} '
                f"({self._primary_field} UInt32, "
                f'{self._vector_field} Array({self.index_param["vector_data_type"]}) CODEC(NONE), '
                f"CONSTRAINT same_length CHECK length(embedding) = {dim}) "
                f"ENGINE = MergeTree() "
                f"ORDER BY {self._primary_field}"
            )

        except Exception as e:
            log.warning(f"Failed to create Clickhouse table: {self.table_name} error: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        pass

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
            # do not iterate for bulk insert
            items = [metadata, embeddings]

            self.conn.insert(
                table=self.table_name,
                data=items,
                column_names=["id", "embedding"],
                column_type_names=["UInt32", f'Array({self.index_param["vector_data_type"]})'],
                column_oriented=True,
            )
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
        parameters = {
            "primary_field": self._primary_field,
            "vector_field": self._vector_field,
            "schema": self.db_config["database"],
            "table": self.table_name,
            "gt": filters.get("id"),
            "k": k,
            "metric_type": self.search_param["metric_type"],
            "query": query,
        }
        if self.case_config.metric_type == "COSINE":
            if filters:
                result = self.conn.query(
                    "SELECT {primary_field:Identifier}, {vector_field:Identifier} "
                    "FROM {schema:Identifier}.{table:Identifier} "
                    "WHERE {primary_field:Identifier} > {gt:UInt32} "
                    "ORDER BY cosineDistance(embedding,{query:Array(Float64)}) "
                    "LIMIT {k:UInt32}",
                    parameters=parameters,
                ).result_rows
                return [int(row[0]) for row in result]

            result = self.conn.query(
                "SELECT {primary_field:Identifier}, {vector_field:Identifier} "
                "FROM {schema:Identifier}.{table:Identifier} "
                "ORDER BY cosineDistance(embedding,{query:Array(Float64)}) "
                "LIMIT {k:UInt32}",
                parameters=parameters,
            ).result_rows
            return [int(row[0]) for row in result]
        if filters:
            result = self.conn.query(
                "SELECT {primary_field:Identifier}, {vector_field:Identifier} "
                "FROM {schema:Identifier}.{table:Identifier} "
                "WHERE {primary_field:Identifier} > {gt:UInt32} "
                "ORDER BY L2Distance(embedding,{query:Array(Float64)}) "
                "LIMIT {k:UInt32}",
                parameters=parameters,
            ).result_rows
            return [int(row[0]) for row in result]

        result = self.conn.query(
            "SELECT {primary_field:Identifier}, {vector_field:Identifier} "
            "FROM {schema:Identifier}.{table:Identifier} "
            "ORDER BY L2Distance(embedding,{query:Array(Float64)}) "
            "LIMIT {k:UInt32}",
            parameters=parameters,
        ).result_rows
        return [int(row[0]) for row in result]
