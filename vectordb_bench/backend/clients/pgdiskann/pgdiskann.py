"""Wrapper around the pg_diskann vector database over VectorDB"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp
from ..api import VectorDB
from .config import PgDiskANNConfigDict, PgDiskANNIndexConfig

log = logging.getLogger(__name__)


class PgDiskANN(VectorDB):
    """Use psycopg instructions"""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _search: sql.Composed  

    def __init__(
        self,
        dim: int,
        db_config: PgDiskANNConfigDict,
        db_case_config: PgDiskANNIndexConfig,
        collection_name: str = "pg_diskann_collection",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "PgDiskANN"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels
        self._scalar_label_field = "label"
        self.where_clause = "" 

        self._index_name = "pgdiskann_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        self.conn, self.cursor = self._create_connection(**self.db_config)

        log.info(f"{self.name} config values: {self.db_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.db_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS pg_diskann CASCADE")
        conn.commit()
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _generate_search_query(self) -> sql.Composed:
        """Generate search query with where_clause placeholder"""
        search_params = self.case_config.search_param()

        if search_params.get("reranking"):
            search_query = sql.SQL(
                """
                SELECT i.id
                FROM (
                    SELECT id, embedding
                    FROM public.{table_name}
                    {where_clause}
                    ORDER BY embedding {metric_fun_op} %s::vector
                    LIMIT {quantized_fetch_limit}::int
                ) i
                ORDER BY i.embedding {reranking_metric_fun_op} %s::vector
                LIMIT %s::int
                """
            ).format(
                table_name=sql.Identifier(self.table_name),
                where_clause=sql.SQL(self.where_clause),
                metric_fun_op=sql.SQL(search_params["metric_fun_op"]),
                reranking_metric_fun_op=sql.SQL(search_params["reranking_metric_fun_op"]),
                quantized_fetch_limit=sql.Literal(search_params["quantized_fetch_limit"]),
            )
        else:
            search_query = sql.Composed(
                [
                    sql.SQL(
                        "SELECT id FROM public.{table_name} {where_clause} ORDER BY embedding "
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        where_clause=sql.SQL(self.where_clause),
                    ),
                    sql.SQL(search_params["metric_fun_op"]),
                    sql.SQL(" %s::vector LIMIT %s::int"),
                ]
            )

        return search_query

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.db_config)

        session_options: dict[str, Any] = self.case_config.session_param()

        if len(session_options) > 0:
            for setting_name, setting_val in session_options.items():
                command = sql.SQL("SET {setting_name} = {setting_val};").format(
                    setting_name=sql.Identifier(setting_name),
                    setting_val=sql.Literal(setting_val),
                )
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            self.conn.commit()

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name),
        )
        log.debug(drop_index_sql.as_string(self.cursor))
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _set_parallel_index_build_param(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        if index_param["maintenance_work_mem"] is not None:
            self.cursor.execute(
                sql.SQL("SET maintenance_work_mem TO {};").format(
                    index_param["maintenance_work_mem"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET maintenance_work_mem TO {};").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["maintenance_work_mem"],
                ),
            )
            self.conn.commit()

        if index_param["max_parallel_workers"] is not None:
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_maintenance_workers TO '{}';").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("SET max_parallel_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_workers TO '{}';").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER TABLE {} SET (parallel_workers = {});").format(
                    sql.Identifier(self.table_name),
                    index_param["max_parallel_workers"],
                ),
            )
            self.conn.commit()

        results = self.cursor.execute(sql.SQL("SHOW max_parallel_maintenance_workers;")).fetchall()
        results.extend(self.cursor.execute(sql.SQL("SHOW max_parallel_workers;")).fetchall())
        results.extend(self.cursor.execute(sql.SQL("SHOW maintenance_work_mem;")).fetchall())
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param: dict[str, Any] = self.case_config.index_param()
        self._set_parallel_index_build_param()

        options = []
        for option_name, option_val in index_param["options"].items():
            if option_val is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option_name),
                        val=sql.Literal(option_val),
                    ),
                )

        with_clause = (
            sql.SQL("WITH ({});").format(sql.SQL(", ").join(options))
            if any(options)
            else sql.Composed(())
        )

        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING {index_type} (embedding {embedding_metric})
            """
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            index_type=sql.Identifier(index_param["index_type"].lower()),
            embedding_metric=sql.Identifier(index_param["metric"]),
        )
        index_create_sql_with_with_clause = (index_create_sql + with_clause).join(" ")
        log.debug(index_create_sql_with_with_clause.as_string(self.cursor))
        self.cursor.execute(index_create_sql_with_with_clause)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            if self.with_scalar_labels:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name} 
                        ({primary_field} BIGINT PRIMARY KEY, embedding vector({dim}), {label_field} VARCHAR(64));
                        """
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                        label_field=sql.Identifier(self._scalar_label_field),
                    ),
                )
            else:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name} 
                        ({primary_field} BIGINT PRIMARY KEY, embedding vector({dim}));
                        """
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                    ),
                )

            self.cursor.execute(
                sql.SQL(
                    "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;"
                ).format(table_name=sql.Identifier(self.table_name)),
            )
            
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create pgdiskann table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if self.with_scalar_labels:
            assert (
                labels_data is not None
            ), "labels_data should be provided if with_scalar_labels is set to True"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            ) as copy:
                if self.with_scalar_labels:
                    copy.set_types(["bigint", "vector", "varchar"])
                    for i, row in enumerate(metadata_arr):
                        copy.write_row((row, embeddings_arr[i], labels_data[i]))
                else:
                    copy.set_types(["bigint", "vector"])
                    for i, row in enumerate(metadata_arr):
                        copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            if kwargs.get("last_batch"):
                self._post_insert()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        """Prepare filter - builds where_clause"""
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for PgDiskANN - {filters}"
            raise ValueError(msg)

        self._search = self._generate_search_query()
        log.debug(f"Search query={self._search.as_string(self.conn)}")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        search_params = self.case_config.search_param()
        q = np.asarray(query)
        
        result = self.cursor.execute(
            self._search,
            (q, q, k) if search_params.get("reranking", False) else (q, k),
            prepare=True,
            binary=True,
        )
        
        return [int(i[0]) for i in result.fetchall()]

