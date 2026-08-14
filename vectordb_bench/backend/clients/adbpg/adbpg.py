"""Wrapper around the Aliyun ADBPG (AnalyticDB for PostgreSQL) vector database."""

import logging
import re
import time
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

NOVA_AUTOTUNE_POLL_SECONDS = 5.0
NOVA_AUTOTUNE_MISSING_PROGRESS_LIMIT = 3
NOVA_AUTOTUNE_NUMERIC_EXPRESSION = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*::\s*(?:real|float4|float8|double\s+precision))?",
    re.IGNORECASE,
)
NOVA_AUTOTUNE_ARRAY_EXPRESSION = re.compile(
    r"ARRAY\s*\[(.*)\](?:\s*::\s*(?:real|float4|float8|double\s+precision)\s*\[\s*\])?",
    re.IGNORECASE | re.DOTALL,
)


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
        self.topk = kwargs.get("k")
        self._validate_autotune_configuration(drop_old)

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
        else:
            # Search-only runs reuse an existing index and skip optimize().
            self._apply_search_setup()

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
                    val=sql.Literal(setting["parameter"]["val"]),
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
        self._apply_search_setup()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _validate_autotune_configuration(self, drop_old: bool) -> None:
        params = self.case_config.autotune_params
        if not params:
            return
        if not drop_old:
            msg = "autotune_param requires a load run that creates a new index"
            raise ValueError(msg)
        if not isinstance(self.topk, int) or isinstance(self.topk, bool) or self.topk <= 0:
            msg = "autotune_param requires a positive benchmark k"
            raise ValueError(msg)

        reserved = {"index_relation", "topk"}.intersection(params)
        if reserved:
            names = ", ".join(sorted(reserved))
            msg = f"autotune_param cannot override {names}; they come from the created index and benchmark k"
            raise ValueError(msg)
        empty = [name for name, value in params.items() if not value.strip()]
        if empty:
            msg = f"autotune_param requires a SQL expression for: {', '.join(sorted(empty))}"
            raise ValueError(msg)
        self._autotune_target_recall_count()

        target_reloptions = {"nova_autotune_topk", "nova_autotune_recall"}.intersection(
            self.case_config.index_reset_reloptions
        )
        if target_reloptions:
            names = ", ".join(sorted(target_reloptions))
            msg = f"Do not combine autotune_param with target-selection index_reset_reloption values: {names}"
            raise ValueError(msg)

    def _autotune_target_recall_count(self) -> int:
        """Return the number of target recalls requested by the SQL expression."""
        expression = self.case_config.autotune_params.get("target_recall")
        if expression is None:
            return 1  # fastann.nova_autotune defaults to one target: 0.99.

        expression = expression.strip()
        if NOVA_AUTOTUNE_NUMERIC_EXPRESSION.fullmatch(expression):
            return 1

        array_match = NOVA_AUTOTUNE_ARRAY_EXPRESSION.fullmatch(expression)
        if array_match:
            values = [value.strip() for value in array_match.group(1).split(",")]
            if values and all(value and NOVA_AUTOTUNE_NUMERIC_EXPRESSION.fullmatch(value) for value in values):
                return len(values)

        msg = (
            "autotune target_recall must be a numeric SQL expression or an ARRAY[...] "
            "of numeric SQL expressions so completion can be validated"
        )
        raise ValueError(msg)

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

    def _apply_search_setup(self) -> None:
        """Run setup SQL, optional autotune, and reloptions on the coordinator."""
        reloptions = self.case_config.index_reset_reloptions
        statements = self.case_config.setup_sql
        autotune_params = self.case_config.autotune_params
        if not reloptions and not statements and not autotune_params:
            return

        connect_config = dict(self.connect_config)
        connect_config.pop("options", None)
        conn, cursor = self._create_connection(**connect_config)
        try:
            # setup_sql is the post-index preparation phase. Commit it before
            # launching the asynchronous worker so ANALYZE and similar changes
            # are visible to autotune.
            for statement in statements:
                log.info("%s setup SQL: %s", self.name, statement)
                cursor.execute(statement)
            if statements:
                conn.commit()

            self._run_nova_autotune(conn, cursor)

            index = sql.Identifier("public", self._index_name)
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
                cursor.execute(
                    sql.SQL("ALTER INDEX {index} SET ({assignments})").format(
                        index=index,
                        assignments=assignments,
                    )
                )
            if reset_options:
                names = sql.SQL(", ").join(sql.Identifier(name) for name in reset_options)
                cursor.execute(sql.SQL("ALTER INDEX {index} RESET ({names})").format(index=index, names=names))
            if set_options or reset_options:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def _run_nova_autotune(self, conn: Connection, cursor: Cursor) -> None:
        """Submit NOVA autotune, then wait for all config rows and worker exit."""
        params = self.case_config.autotune_params
        if not params:
            return

        expected_config_count = self._autotune_target_recall_count()
        # This is bound as a query parameter, not interpolated into SQL. ADBPG
        # currently derives index names from framework-controlled table and
        # algorithm names, so the regclass text form is stable.
        qualified_index = f"public.{self._index_name}"
        arguments = [
            sql.SQL("index_relation => %s::text"),
            sql.SQL("topk => %s::integer"),
        ]
        arguments.extend(
            sql.SQL("{name} => {value}").format(
                name=sql.Identifier(name),
                value=sql.SQL(value),
            )
            for name, value in params.items()
        )
        submit_sql = sql.SQL("SELECT fastann.nova_autotune({arguments})").format(
            arguments=sql.SQL(", ").join(arguments)
        )
        cursor.execute(submit_sql, (qualified_index, self.topk))
        row = cursor.fetchone()
        if row is None:
            msg = "nova_autotune did not return a task handle"
            raise RuntimeError(msg)
        handle = int(row[0])
        conn.commit()
        log.info(
            "NOVA_AUTOTUNE_SUBMITTED handle=%s index=%s topk=%s params=%s",
            handle,
            qualified_index,
            self.topk,
            params,
        )

        deadline = time.monotonic() + self.case_config.autotune_timeout
        missing_progress_polls = 0
        latest_progress = None
        latest_config_count = 0
        while True:
            cursor.execute(
                """
                SELECT pid, stage, work_done, work_total, target_count
                FROM fastann.nova_autotune_progress(%s)
                """,
                (handle,),
            )
            latest_progress = cursor.fetchone()
            cursor.execute(
                """
                SELECT count(*)
                FROM fastann.nova_autotune_configs c
                JOIN pg_catalog.pg_class i
                  ON i.oid = c.index_relid
                 AND i.relfilenode::oid = c.index_relfilenode
                WHERE i.oid = %s::regclass
                  AND c.topk = %s
                """,
                (qualified_index, self.topk),
            )
            count_row = cursor.fetchone()
            latest_config_count = int(count_row[0]) if count_row is not None else 0
            conn.commit()

            if latest_progress is None and latest_config_count == expected_config_count:
                log.info(
                    "NOVA_AUTOTUNE_COMPLETED handle=%s index=%s configs=%s",
                    handle,
                    qualified_index,
                    latest_config_count,
                )
                return

            if latest_progress is None:
                missing_progress_polls += 1
                if missing_progress_polls >= NOVA_AUTOTUNE_MISSING_PROGRESS_LIMIT:
                    msg = (
                        f"nova_autotune handle {handle} finished with {latest_config_count} config rows "
                        f"for topk={self.topk}; expected {expected_config_count}"
                    )
                    raise RuntimeError(msg)
            else:
                missing_progress_polls = 0

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = (
                    f"nova_autotune handle {handle} exceeded {self.case_config.autotune_timeout}s; "
                    f"progress={latest_progress}, configs={latest_config_count}/{expected_config_count}"
                )
                raise TimeoutError(msg)
            log.info(
                "NOVA_AUTOTUNE_PROGRESS handle=%s progress=%s configs=%s/%s",
                handle,
                latest_progress,
                latest_config_count,
                expected_config_count,
            )
            time.sleep(min(NOVA_AUTOTUNE_POLL_SECONDS, remaining))

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
