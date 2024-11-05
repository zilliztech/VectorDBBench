"""Wrapper around the Pgvector vector database over VectorDB"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from ..api import VectorDB
from .config import PgVectorConfigDict, PgVectorIndexConfig

log = logging.getLogger(__name__)


class PgVector(VectorDB):
    """Use psycopg instructions"""

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _filtered_search: sql.Composed
    _unfiltered_search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: PgVectorConfigDict,
        db_case_config: PgVectorIndexConfig,
        collection_name: str = "pg_vector_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PgVector"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "pgvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # create vector extension
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
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

    def _generate_search_query(self, filtered: bool = False) -> sql.Composed:
        index_param = self.case_config.index_param()
        reranking = self.case_config.search_param()["reranking"]
        column_name = (
            sql.SQL("binary_quantize({0})").format(sql.Identifier("embedding"))
            if index_param["quantization_type"] == "bit" and index_param["table_quantization_type"] != "bit"
            else sql.SQL("embedding")
        )
        search_vector = (
            sql.SQL("binary_quantize({0})").format(sql.Placeholder())
            if index_param["quantization_type"] == "bit"
            else sql.Placeholder()
        )

        # The following sections assume that the quantization_type value matches the quantization function name
        if index_param["quantization_type"] != index_param["table_quantization_type"]:
            # Reranking makes sense only if table quantization is not "bit"
            if index_param["quantization_type"] == "bit" and reranking:
                # Embeddings needs to be passed to binary_quantize function if quantization_type is bit
                search_query = sql.Composed(
                    [
                        sql.SQL(
                            """
                            SELECT i.id
                            FROM (
                                SELECT id, embedding {reranking_metric_fun_op} %s::{table_quantization_type} AS distance
                                FROM public.{table_name} {where_clause}
                                ORDER BY {column_name}::{quantization_type}({dim})
                            """,
                        ).format(
                            table_name=sql.Identifier(self.table_name),
                            column_name=column_name,
                            reranking_metric_fun_op=sql.SQL(
                                self.case_config.search_param()["reranking_metric_fun_op"],
                            ),
                            search_vector=search_vector,
                            table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                            where_clause=sql.SQL("WHERE id >= %s") if filtered else sql.SQL(""),
                        ),
                        sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                        sql.SQL(
                            """
                                {search_vector}::{quantization_type}({dim})
                                LIMIT {quantized_fetch_limit}
                            ) i
                            ORDER BY i.distance
                            LIMIT %s::int
                            """,
                        ).format(
                            search_vector=search_vector,
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                            quantized_fetch_limit=sql.Literal(
                                self.case_config.search_param()["quantized_fetch_limit"],
                            ),
                        ),
                    ],
                )
            else:
                search_query = sql.Composed(
                    [
                        sql.SQL(
                            """
                            SELECT id FROM public.{table_name}
                            {where_clause} ORDER BY {column_name}::{quantization_type}({dim})
                            """,
                        ).format(
                            table_name=sql.Identifier(self.table_name),
                            column_name=column_name,
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                            where_clause=sql.SQL("WHERE id >= %s") if filtered else sql.SQL(""),
                        ),
                        sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                        sql.SQL(" {search_vector}::{quantization_type}({dim}) LIMIT %s::int").format(
                            search_vector=search_vector,
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                        ),
                    ]
                )
        else:
            search_query = sql.Composed(
                [
                    sql.SQL(
                        "SELECT id FROM public.{table_name} {where_clause} ORDER BY embedding ",
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        where_clause=sql.SQL("WHERE id >= %s") if filtered else sql.SQL(""),
                    ),
                    sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                    sql.SQL(" {search_vector}::{quantization_type}({dim}) LIMIT %s::int").format(
                        search_vector=search_vector,
                        quantization_type=sql.SQL(index_param["quantization_type"]),
                        dim=sql.Literal(self.dim),
                    ),
                ]
            )

        return search_query

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """

        self.conn, self.cursor = self._create_connection(**self.db_config)

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

        self._filtered_search = self._generate_search_query(filtered=True)
        self._unfiltered_search = self._generate_search_query()

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

        index_param = self.case_config.index_param()
        self._set_parallel_index_build_param()
        options = []
        for option in index_param["index_creation_with_options"]:
            if option["val"] is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option["option_name"]),
                        val=sql.Identifier(str(option["val"])),
                    ),
                )
        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if any(options) else sql.Composed(())

        if index_param["quantization_type"] != index_param["table_quantization_type"]:
            index_create_sql = sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
                USING {index_type} (({column_name}::{quantization_type}({dim})) {embedding_metric})
                """,
            ).format(
                index_name=sql.Identifier(self._index_name),
                table_name=sql.Identifier(self.table_name),
                column_name=(
                    sql.SQL("binary_quantize({0})").format(sql.Identifier("embedding"))
                    if index_param["quantization_type"] == "bit"
                    else sql.Identifier("embedding")
                ),
                index_type=sql.Identifier(index_param["index_type"]),
                # This assumes that the quantization_type value matches the quantization function name
                quantization_type=sql.SQL(index_param["quantization_type"]),
                dim=self.dim,
                embedding_metric=sql.Identifier(index_param["metric"]),
            )
        else:
            index_create_sql = sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
                USING {index_type} (embedding {embedding_metric})
                """,
            ).format(
                index_name=sql.Identifier(self._index_name),
                table_name=sql.Identifier(self.table_name),
                index_type=sql.Identifier(index_param["index_type"]),
                embedding_metric=sql.Identifier(index_param["metric"]),
            )

        index_create_sql_with_with_clause = (index_create_sql + with_clause).join(" ")
        log.debug(index_create_sql_with_with_clause.as_string(self.cursor))
        self.cursor.execute(index_create_sql_with_with_clause)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            # create table
            self.cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS public.{table_name}
                    (id BIGINT PRIMARY KEY, embedding {table_quantization_type}({dim}));
                    """
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                    dim=dim,
                )
            )
            self.cursor.execute(
                sql.SQL(
                    "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
                ).format(table_name=sql.Identifier(self.table_name)),
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            if index_param["table_quantization_type"] == "bit":
                with self.cursor.copy(
                    sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT TEXT)").format(
                        table_name=sql.Identifier(self.table_name)
                    )
                ) as copy:
                    # Same logic as pgvector binary_quantize
                    for i, row in enumerate(metadata_arr):
                        embeddings_bit = ""
                        for embedding in embeddings_arr[i]:
                            if embedding > 0:
                                embeddings_bit += "1"
                            else:
                                embeddings_bit += "0"
                        copy.write_row((str(row), embeddings_bit))
            else:
                with self.cursor.copy(
                    sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                        table_name=sql.Identifier(self.table_name)
                    )
                ) as copy:
                    if index_param["table_quantization_type"] == "halfvec":
                        copy.set_types(["bigint", "halfvec"])
                        for i, row in enumerate(metadata_arr):
                            copy.write_row((row, np.float16(embeddings_arr[i])))
                    else:
                        copy.set_types(["bigint", "vector"])
                        for i, row in enumerate(metadata_arr):
                            copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            if kwargs.get("last_batch"):
                self._post_insert()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()
        search_param = self.case_config.search_param()
        q = np.asarray(query)
        if filters:
            gt = filters.get("id")
            if index_param["quantization_type"] == "bit" and search_param["reranking"]:
                result = self.cursor.execute(
                    self._filtered_search,
                    (q, gt, q, k),
                    prepare=True,
                    binary=True,
                )
            else:
                result = self.cursor.execute(
                    self._filtered_search,
                    (gt, q, k),
                    prepare=True,
                    binary=True,
                )

        elif index_param["quantization_type"] == "bit" and search_param["reranking"]:
            result = self.cursor.execute(
                self._unfiltered_search,
                (q, q, k),
                prepare=True,
                binary=True,
            )
        else:
            result = self.cursor.execute(
                self._unfiltered_search,
                (q, k),
                prepare=True,
                binary=True,
            )

        return [int(i[0]) for i in result.fetchall()]
