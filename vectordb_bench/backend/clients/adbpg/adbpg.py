"""Wrapper around the Aliyun ADBPG (AnalyticDB for PostgreSQL) vector database."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from copy import copy
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql
from psycopg.types.json import Jsonb

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AdbpgConfigDict, AdbpgIndexConfig

log = logging.getLogger(__name__)

UDF_CLEANUP_GRACE_SECONDS = 30


class AdbpgTimeoutError(RuntimeError):
    """ADBPG canceled a database statement after its configured timeout."""


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
        # Index name derives from the table name + algorithm, e.g. vector_1024d_10m_novamr_index.
        self._index_name = f"{self.table_name}_{self.case_config.algorithm}_index"

        self.where_clause = ""

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        log.info("%s case config: %s", self.name, self.case_config)
        try:
            if drop_old:
                self._check_required_udfs()
                self._drop_index()
                self._drop_table()
                self._create_table(dim)
            else:
                self._apply_search_reloptions()
        finally:
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
        """Open a database session, yield, then close it."""
        self.conn, self.cursor = self._create_connection(**self.connect_config)

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
        payload = self._build_udf_payload()
        self._call_udf("vectordbbench_build", payload)
        try:
            autotune = self.case_config.autotune_parameters
            statement_timeout = autotune.timeout + UDF_CLEANUP_GRACE_SECONDS if autotune is not None else None
            self._call_udf("vectordbbench_optimize", payload, timeout=statement_timeout)
            self._apply_search_reloptions()
        except Exception:
            try:
                self.conn.rollback()
                self._drop_index()
            except Exception:
                log.exception("Failed to remove incomplete ADBPG index %s", self._index_name)
            raise

    def _check_required_udfs(self) -> None:
        assert self.cursor is not None, "Cursor is not initialized"
        signatures = (
            "fastann.vectordbbench_build(jsonb)",
            "fastann.vectordbbench_optimize(jsonb)",
        )
        row = self.cursor.execute(
            """
            SELECT to_regprocedure(%s), has_function_privilege(to_regprocedure(%s), 'EXECUTE'),
                   to_regprocedure(%s), has_function_privilege(to_regprocedure(%s), 'EXECUTE'),
                   has_schema_privilege(to_regnamespace('fastann'), 'USAGE')
            """,
            (signatures[0], signatures[0], signatures[1], signatures[1]),
        ).fetchone()
        access = ((row[0], row[1]), (row[2], row[3]))
        unavailable = [
            signature
            for signature, (oid, allowed) in zip(signatures, access, strict=True)
            if oid is None or not allowed
        ]
        if unavailable or not row[-1]:
            details = list(unavailable)
            if not row[-1]:
                details.append("USAGE on schema fastann")
            msg = f"Required ADBPG UDF access is unavailable: {', '.join(details)}"
            raise RuntimeError(msg)

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {schema}.{index_name}").format(
            schema=sql.Identifier("public"),
            index_name=sql.Identifier(self._index_name),
        )
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _build_udf_payload(self) -> dict[str, Any]:
        columns = [
            {"name": self._primary_field, "type": "bigint", "role": "primary_key"},
            {"name": self._vector_field, "type": f"vector({self.dim})", "role": "vector"},
        ]
        if self.with_scalar_labels:
            columns.append(
                {"name": self._scalar_label_field, "type": "varchar(64)", "role": "filter"},
            )
        parameters = {
            "build": self.case_config.build_parameters.model_dump(mode="json"),
            "search": self.case_config.search_parameters.model_dump(mode="json"),
        }
        if self.case_config.autotune_parameters is not None:
            parameters["autotune"] = self.case_config.autotune_parameters.model_dump(mode="json")
        return {
            "api_version": 1,
            "relation": {
                "schema": "public",
                "table": self.table_name,
                "index": self._index_name,
            },
            "columns": columns,
            "metric": self.case_config.parse_metric(),
            "parameters": parameters,
        }

    def _call_udf(self, function_name: str, payload: dict[str, Any], timeout: int | None = None) -> list[Any]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        qualified_name = f"fastann.{function_name}"
        command = sql.SQL("SELECT * FROM {schema}.{function}(%s::jsonb)").format(
            schema=sql.Identifier("fastann"),
            function=sql.Identifier(function_name),
        )
        try:
            if timeout is not None:
                self.cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{timeout}s",),
                )
            rows = self.cursor.execute(command, (Jsonb(payload),)).fetchall()
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            cancel_reason = getattr(getattr(exc, "diag", None), "message_primary", None) or str(exc)
            if (
                timeout is not None
                and isinstance(exc, psycopg.errors.QueryCanceled)
                and "statement timeout" in cancel_reason.lower()
            ):
                msg = f"{qualified_name} timed out after {timeout} seconds"
                raise AdbpgTimeoutError(msg) from exc
            msg = f"{qualified_name} failed: {exc}"
            raise RuntimeError(msg) from exc
        else:
            log.info("%s returned diagnostic rows: %s", qualified_name, rows)
            return rows

    def _apply_search_reloptions(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        reloptions = self.case_config.search_parameters.reloption
        set_options = [(name, value) for name, value in reloptions.items() if value is not None]
        reset_options = [name for name, value in reloptions.items() if value is None]
        if set_options:
            assignments = sql.SQL(", ").join(
                sql.SQL("{name} = {value}").format(
                    name=sql.Identifier(name),
                    value=sql.Literal(value),
                )
                for name, value in set_options
            )
            command = sql.SQL("ALTER INDEX {schema}.{index} SET ({assignments})").format(
                schema=sql.Identifier("public"),
                index=sql.Identifier(self._index_name),
                assignments=assignments,
            )
            self.cursor.execute(command)
        if reset_options:
            names = sql.SQL(", ").join(sql.Identifier(name) for name in reset_options)
            command = sql.SQL("ALTER INDEX {schema}.{index} RESET ({names})").format(
                schema=sql.Identifier("public"),
                index=sql.Identifier(self._index_name),
                names=names,
            )
            self.cursor.execute(command)
        if set_options or reset_options:
            self.conn.commit()

    def _apply_search_gucs(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        search_gucs = self.case_config.search_parameters.guc
        for name, value in search_gucs.items():
            if value is None:
                command = sql.SQL("RESET {setting_name}").format(setting_name=sql.Identifier(name))
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            else:
                self.cursor.execute(
                    "SELECT set_config(%s, %s, false)",
                    (name, str(value)),
                )
        if search_gucs:
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

        self._apply_search_gucs()
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
