"""Wrapper around the Lakebase vector database over VectorDB"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import LakebaseANNConfig, LakebaseVectorConfigDict

log = logging.getLogger(__name__)


class LakebaseVector(VectorDB):
    """Use psycopg instructions"""

    thread_safe: bool = False
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
        db_config: LakebaseVectorConfigDict,
        db_case_config: LakebaseANNConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "LakebaseVector"
        self.case_config = db_case_config
        self.table_name = db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels
        self._index_name = "lakebase_vector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # create lakebase_vector extension
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
        self.conn.commit()

        log.info(f"{self.name} config values: {self.connect_config}\n{self.case_config}")
        if db_case_config.create_index_before_load or not db_case_config.create_index_after_load:
            msg = "LakebaseVector supports only create_index_after_load"
            log.error(msg)
            raise RuntimeError(msg)
        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]

        if len(session_options) > 0:
            for setting in session_options:
                command = sql.SQL("SET {setting_name} " + "= {val};").format(
                    setting_name=sql.Identifier(setting["parameter"]["setting_name"]),
                    val=sql.Identifier(str(setting["parameter"]["val"])),
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

        if index_param["max_parallel_workers"] is not None:
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_maintenance_workers TO '{}';").format(
                    sql.Identifier(self.connect_config["user"]),
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
                    sql.Identifier(self.connect_config["user"]),
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
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param = self.case_config.index_param()
        self._set_parallel_index_build_param()
        index_create_sql = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name} "
            "USING {index_type} ({vector_field} {metric})"
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            index_type=sql.SQL(index_param["index_type"]),
            vector_field=sql.Identifier(self._vector_field),
            metric=sql.SQL(index_param["metric"]),
        )
        log.debug(index_create_sql.as_string(self.cursor))
        self.cursor.execute(index_create_sql)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # create table
        if self.with_scalar_labels:
            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{table_name} "
                    "({primary_field} BIGINT PRIMARY KEY, {vector_field} vector({dim}), "
                    "{label_field} VARCHAR(64))"
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    vector_field=sql.Identifier(self._vector_field),
                    dim=sql.Literal(dim),
                    label_field=sql.Identifier(self._scalar_label_field),
                )
            )
        else:
            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{table_name} "
                    "({primary_field} BIGINT PRIMARY KEY, {vector_field} vector({dim}))"
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    vector_field=sql.Identifier(self._vector_field),
                    dim=sql.Literal(dim),
                )
            )
        self.cursor.execute(
            sql.SQL("ALTER TABLE public.{table_name} ALTER COLUMN {vector_field} SET STORAGE PLAIN").format(
                table_name=sql.Identifier(self.table_name),
                vector_field=sql.Identifier(self._vector_field),
            )
        )
        self.conn.commit()

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
            assert labels_data is not None, "labels_data should be provided if with_scalar_labels is set to True"
        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)
            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                )
            ) as copy:
                for i, row in enumerate(metadata_arr):
                    if self.with_scalar_labels:
                        copy.set_types(["bigint", "vector", "varchar"])
                        copy.write_row((row, embeddings_arr[i], labels_data[i]))
                    else:
                        copy.set_types(["bigint", "vector"])
                        copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into lakebase_vector table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            where_clause = ""
        elif filters.type == FilterOp.NumGE:
            where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for lakebase_vector - {filters}"
            raise ValueError(msg)
        self._search = sql.Composed(
            [
                sql.SQL(
                    "SELECT {primary_field} FROM public.{table_name} {where_clause} ORDER BY {vector_field} "
                ).format(
                    primary_field=sql.Identifier(self._primary_field),
                    table_name=sql.Identifier(self.table_name),
                    where_clause=sql.SQL(where_clause),
                    vector_field=sql.Identifier(self._vector_field),
                ),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ]
        )

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        result = self.cursor.execute(self._search, (np.asarray(query), k), prepare=True, binary=True)
        return [int(i[0]) for i in result.fetchall()]
