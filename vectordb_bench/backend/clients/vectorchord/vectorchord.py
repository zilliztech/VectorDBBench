"""Wrapper around the VectorChord vector database over VectorDB"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from ...filter import Filter, FilterOp
from ..api import VectorDB
from .config import VectorChordConfigDict, VectorChordIndexConfig

log = logging.getLogger(__name__)


class VectorChord(VectorDB):
    """Use psycopg instructions"""

    thread_safe: bool = False
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
    ]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _search: sql.Composed
    where_clause: str = ""

    def __init__(
        self,
        dim: int,
        db_config: VectorChordConfigDict,
        db_case_config: VectorChordIndexConfig,
        collection_name: str = "vectorchord_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "VectorChord"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "vectorchord_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        index_param = self.case_config.index_param()
        self._quantization_type = index_param["quantization_type"]
        self._index_method = index_param["index_type"]

        self.conn, self.cursor = self._create_connection(**self.db_config)

        # create vectorchord extension if not exists
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        self.conn.commit()

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
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()
        
        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: dict[str, Any] = self.case_config.session_param()

        if len(session_options) > 0:
            for setting_name, setting_val in session_options.items():
                command = sql.SQL("SET {setting_name} " + "= {setting_val};").format(
                    setting_name=sql.Identifier(setting_name),
                    setting_val=sql.Literal(str(setting_val)),
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
                sql.SQL("SET max_parallel_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
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

        results = self.cursor.execute(sql.SQL("SHOW max_parallel_workers;")).fetchall()
        results.extend(self.cursor.execute(sql.SQL("SHOW max_parallel_maintenance_workers;")).fetchall())
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param: dict[str, Any] = self.case_config.index_param()
        self._set_parallel_index_build_param()

        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING {index_method} (embedding {embedding_metric})
            """,
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            index_method=sql.SQL(self._index_method),
            embedding_metric=sql.Identifier(index_param["metric"]),
        )

        options_str = index_param.get("options", "")
        if options_str:
            with_clause = sql.SQL(
                "WITH (options = $vchord$\n{options}\n$vchord$);",
            ).format(options=sql.SQL(options_str))
        else:
            with_clause = sql.SQL(";")

        full_sql = index_create_sql + sql.SQL(" ") + with_clause
        log.debug(full_sql.as_string(self.cursor))
        self.cursor.execute(full_sql)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            col_type = self._quantization_type
            if col_type in ("rabitq8", "rabitq4"):
                # rabitq types need vector column + quantization during insert
                col_type = "vector"

            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{table_name} "
                    "(id BIGINT PRIMARY KEY, embedding {col_type}({dim}));",
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    col_type=sql.SQL(col_type),
                    dim=dim,
                ),
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create vectorchord table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            if self._quantization_type == "halfvec":
                with self.cursor.copy(
                    sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                        table_name=sql.Identifier(self.table_name),
                    ),
                ) as copy:
                    copy.set_types(["bigint", "halfvec"])
                    for i, row in enumerate(metadata_arr):
                        copy.write_row((row, np.float16(embeddings_arr[i])))
            else:
                # vector, rabitq8, rabitq4 all store as vector column
                with self.cursor.copy(
                    sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                        table_name=sql.Identifier(self.table_name),
                    ),
                ) as copy:
                    copy.set_types(["bigint", "vector"])
                    for i, row in enumerate(metadata_arr):
                        copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into vectorchord table ({self.table_name}), error: {e}")
            return 0, e

    def _generate_search_query(self) -> sql.Composed:
        # Search query cast type: rabitq8/rabitq4 queries still accept ::vector input
        cast_type = "vector"
        return sql.Composed(
            [
                sql.SQL("SELECT id FROM public.{table_name} {where_clause} ORDER BY embedding ").format(
                    table_name=sql.Identifier(self.table_name),
                    where_clause=sql.SQL(self.where_clause),
                ),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(f" %s::{cast_type} LIMIT %s::int"),
            ],
        )

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        else:
            msg = f"Not support Filter for VectorChord - {filters}"
            raise ValueError(msg)

        self._search = self._generate_search_query()

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = np.asarray(query)
        result = self.cursor.execute(self._search, (q, k), prepare=True, binary=True)
        return [int(i[0]) for i in result.fetchall()]
