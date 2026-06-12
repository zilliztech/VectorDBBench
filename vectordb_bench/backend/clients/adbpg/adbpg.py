"""Wrapper around the Aliyun ADBPG (AnalyticDB for PostgreSQL) vector database."""

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
from .config import AdbpgConfigDict, AdbpgIndexConfig

log = logging.getLogger(__name__)


class Adbpg(VectorDB):
    """ADBPG vector database client, using psycopg."""

    # psycopg Cursor is not thread-safe and the COPY protocol cannot be
    # interleaved on a shared connection. Match PgVector/VectorChord and
    # let ConcurrentInsertRunner clamp max_workers=1.
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
        db_config: AdbpgConfigDict,
        db_case_config: AdbpgIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Adbpg"
        self.case_config = db_case_config
        # Allow the framework layer (task_runner) to inject a case-specific table
        # name via the `collection_name` kwarg (see Doris for the same pattern).
        override_name = kwargs.get("collection_name")
        self.table_name = override_name if override_name else db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"
        # Index name derives from the table name + algorithm, e.g. vector_1024d_10m_novam_index.
        self._index_name = f"{self.table_name}_{self.case_config.algorithm}_index"

        self.where_clause = ""

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        log.info(f"{self.name} config values: {self.connect_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.connect_config}\n{self.case_config}"
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

    def _generate_search_query(self) -> sql.Composed:
        search_param = self.case_config.search_param()
        distance_operator = {
            "l2": "<->",
            "ip": "<#>",
            "cosine": "<=>",
        }.get(search_param["metric"], "<->")

        where_clause = sql.SQL(self.where_clause) if self.where_clause else sql.SQL("")

        return sql.Composed(
            [
                sql.SQL(
                    "SELECT {primary_field} FROM public.{table_name} {where_clause} ORDER BY {vector_field} ",
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    where_clause=where_clause,
                    vector_field=sql.Identifier(self._vector_field),
                ),
                sql.SQL(distance_operator),
                sql.SQL(" {search_vector}::vector({dim}) LIMIT %s::int").format(
                    search_vector=sql.Placeholder(),
                    dim=self.dim,
                ),
            ],
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Open a session, apply GUCs, yield, then close."""
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]

        if len(session_options) > 0:
            for setting in session_options:
                command = sql.SQL("SET {setting_name} = {val};").format(
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

        if index_param["build_parallel_processes"] is not None:
            self.cursor.execute(
                sql.SQL("SET fastann.build_parallel_processes TO {};").format(
                    index_param["build_parallel_processes"],
                ),
            )
            self.conn.commit()

        results = self.cursor.execute(sql.SQL("SHOW fastann.build_parallel_processes;")).fetchall()
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param = self.case_config.index_param()
        self._set_parallel_index_build_param()

        # Pre-build GUC: raise optimizer level before creating the ANN index.
        self.cursor.execute(sql.SQL("SET fastann.nova_build_optimize_level = 3;"))
        self.conn.commit()

        options = []
        options.append(sql.SQL("dim = {dim}").format(dim=sql.Literal(self.dim)))
        options.append(
            sql.SQL("distancemeasure = {measure}").format(
                measure=sql.Identifier(index_param["metric"]),
            ),
        )

        for option in index_param["index_creation_with_options"]:
            if option["val"] is not None:
                # When `raw` is set, emit the value as a bare SQL token
                # (e.g. auto_reduction=on) instead of a quoted literal.
                rendered_val = sql.SQL(str(option["val"])) if option.get("raw") else sql.Literal(option["val"])
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option["option_name"]),
                        val=rendered_val,
                    ),
                )

        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if options else sql.Composed(())

        # Covering index: always INCLUDE the primary field (e.g. id).
        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING ann ({vector_field}) INCLUDE ({primary_field})
            """,
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            vector_field=sql.Identifier(self._vector_field),
            primary_field=sql.Identifier(self._primary_field),
        )

        full_sql = (index_create_sql + with_clause).join(" ")
        log.debug(full_sql.as_string(self.cursor))
        self.cursor.execute(full_sql)
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
                        """,
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        dim=dim,
                        label_field=sql.Identifier(self._scalar_label_field),
                    ),
                )
            else:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding vector({dim}));
                        """,
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        dim=dim,
                    ),
                )

            self.cursor.execute(
                sql.SQL(
                    "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
                ).format(table_name=sql.Identifier(self.table_name)),
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create adbpg table: {self.table_name} error: {e}")
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
            assert labels_data is not None, "labels_data should be provided if with_scalar_labels is set to True"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
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
            log.warning(f"Failed to insert data into adbpg table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Adbpg - {filters}"
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
        result = self.cursor.execute(
            self._search,
            (q, k),
            prepare=True,
            binary=True,
        )
        return [int(i[0]) for i in result.fetchall()]
