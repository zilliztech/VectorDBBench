"""Wrapper around the SVS vector database over VectorDB"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import copy
from typing import Any

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import MetricType, VectorDB
from .config import PgSVSConfigDict, PgSVSIndexConfig, PgSVSVamanaConfig

log = logging.getLogger(__name__)


def _safe_connect_config(connect_config: dict[str, Any]) -> dict[str, Any]:
    # allow-list: only ever surfaces these named fields, so a future field added
    # to connect_config can't leak into logs by default.
    return {k: connect_config[k] for k in ("host", "port", "dbname", "user") if k in connect_config}


class PgSVS(VectorDB):
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
        db_config: PgSVSConfigDict,
        db_case_config: PgSVSIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "PgSVS"
        self.case_config = db_case_config
        self.table_name = db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._index_name = f"{self.table_name}_svs_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # svs is a superuser-only C extension; verify it's pre-installed rather than attempting CREATE
        installed = self.cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'svs'").fetchone()
        if not installed:
            raise RuntimeError(
                "Extension 'svs' is not installed. Run 'CREATE EXTENSION svs;' as a PostgreSQL superuser first."
            )

        safe_connect_config = _safe_connect_config(self.connect_config)
        log.info(f"{self.name} config values: {safe_connect_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {safe_connect_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            # DROP TABLE cascades and drops this table's own index too; no separate DROP INDEX needed.
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()
        else:
            self._validate_existing_index(dim)

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

    def copy_for_thread(self) -> "VectorDB":
        # psycopg holds an open socket that can't be deep-copied; shallow-copy
        # and drop the connection so init() reconnects inside the worker thread.
        db_copy = copy(self)
        db_copy.conn = None
        db_copy.cursor = None
        return db_copy

    def _generate_search_query(self) -> sql.Composed:
        index_param = self.case_config.index_param()
        # vamana has no bit opclass (rejected by PgSVSVamanaConfig.reject_bit_quantization), so
        # quantization_type can never be "bit" here -- always the plain (non-binary_quantize) path.
        column_name = sql.SQL(self._vector_field)
        search_vector = sql.Placeholder()

        if index_param["quantization_type"] != index_param["table_quantization_type"]:
            search_query = sql.Composed(
                [
                    sql.SQL(
                        """
                        SELECT {primary_field} FROM public.{table_name}
                        {where_clause} ORDER BY {column_name}::{quantization_type}({dim})
                        """,
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        column_name=column_name,
                        quantization_type=sql.SQL(index_param["quantization_type"]),
                        dim=sql.Literal(self.dim),
                        where_clause=sql.SQL(self.where_clause),
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
                        "SELECT {primary_field} FROM public.{table_name} {where_clause} ORDER BY {vector_field}",
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        vector_field=sql.Identifier(self._vector_field),
                        where_clause=sql.SQL(self.where_clause),
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

    def _validate_existing_index(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # to_regclass returns NULL (rather than erroring) if the table doesn't exist yet,
        # which the join below then naturally turns into zero rows.
        row = self.cursor.execute(
            sql.SQL(
                """
                SELECT am.amname, opc.opcname, format_type(att.atttypid, att.atttypmod) AS coltype,
                       idxc.reloptions
                FROM pg_index i
                JOIN pg_class idxc ON idxc.oid = i.indexrelid
                JOIN pg_am am ON am.oid = idxc.relam
                JOIN pg_opclass opc ON opc.oid = i.indclass[0]
                JOIN pg_attribute att ON att.attrelid = i.indrelid AND att.attnum = i.indkey[0]
                WHERE idxc.relname = {index_name} AND i.indrelid = to_regclass({table_name})
                """,
            ).format(
                index_name=sql.Literal(self._index_name),
                table_name=sql.Literal(f"public.{self.table_name}"),
            ),
        ).fetchone()
        if row is None:
            # No existing index to reuse (table and/or index not created yet); it'll be
            # built normally via create_index_before_load/create_index_after_load.
            return

        actual_am, actual_opc, actual_coltype, actual_reloptions = row
        reloptions = dict(opt.split("=", 1) for opt in (actual_reloptions or []))

        index_param = self.case_config.index_param()
        expected_coltype = f"{index_param['table_quantization_type']}({dim})"
        expected_options = {opt["option_name"]: opt["val"] for opt in index_param["index_creation_with_options"]}

        mismatches = []
        if actual_am != "vamana":
            mismatches.append(f"access_method={actual_am} (expected vamana)")
        if actual_opc != index_param["metric"]:
            mismatches.append(f"operator_class={actual_opc} (expected {index_param['metric']})")
        if actual_coltype != expected_coltype:
            mismatches.append(f"column_type={actual_coltype} (expected {expected_coltype})")
        for option_name, expected_val in expected_options.items():
            actual_val = reloptions.get(option_name)
            if actual_val != expected_val:
                mismatches.append(f"{option_name}={actual_val} (expected {expected_val})")

        if mismatches:
            msg = f"Existing index {self._index_name} doesn't match this run's config: {', '.join(mismatches)}"
            raise RuntimeError(msg)

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
            self.conn.commit()

        if index_param["max_parallel_workers"] is not None:
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("SET max_parallel_workers TO '{}';").format(
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
                # vamana has no bit opclass, so quantization_type can never be "bit" here.
                column_name=sql.Identifier("embedding"),
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
            if self.with_scalar_labels:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding {table_quantization_type}({dim}), {label_field} VARCHAR(64));
                        """,  # noqa: E501
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                        label_field=sql.Identifier(self._scalar_label_field),
                    )
                )
            else:
                self.cursor.execute(
                    sql.SQL("""
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding {table_quantization_type}({dim}));
                        """).format(
                        table_name=sql.Identifier(self.table_name),
                        table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                    )
                )

            self.cursor.execute(
                sql.SQL(
                    "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
                ).format(table_name=sql.Identifier(self.table_name)),
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create SVS table: {self.table_name} error: {e}")
            raise e from None

    def need_normalize_cosine(self) -> bool:
        if (
            isinstance(self.case_config, PgSVSVamanaConfig)
            and self.case_config.normalize_cosine
            and self.case_config.metric_type == MetricType.COSINE
        ):
            log.info("current SVS config enables normalization for cosine metric")
            return True
        return False

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

        index_param = self.case_config.index_param()

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            # vamana has no bit opclass, so table_quantization_type can never be "bit" here.
            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name)
                )
            ) as copy:
                if index_param["table_quantization_type"] == "halfvec":
                    for i, row in enumerate(metadata_arr):
                        if self.with_scalar_labels:
                            copy.set_types(["bigint", "halfvec", "varchar"])
                            copy.write_row((row, HalfVector(embeddings_arr[i]), labels_data[i]))
                        else:
                            copy.set_types(["bigint", "halfvec"])
                            copy.write_row((row, HalfVector(embeddings_arr[i])))
                else:
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
            log.warning(f"Failed to insert data into SVS table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for PgSVS - {filters}"
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

        configured_window = self.case_config.search_window_size
        if configured_window is not None and k > configured_window:
            msg = (
                f"k={k} exceeds search_window_size={configured_window}; "
                "search_window_size needs to be at least k for meaningful accuracy."
            )
            raise ValueError(msg)

        q = np.asarray(query)
        result = self.cursor.execute(
            self._search,
            (q, k),
            prepare=True,
            binary=True,
        )
        return [int(i[0]) for i in result.fetchall()]
