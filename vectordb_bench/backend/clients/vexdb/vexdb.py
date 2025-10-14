"""Wrapper around the VexDB vector database over VectorDB"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from psycopg.types import TypeInfo
from psycopg import Connection, Cursor, sql
from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
from pgvector.psycopg import Vector

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import VexDBConfigDict, VexDBIndexConfig

log = logging.getLogger(__name__)

class VectorDumper(Dumper):
    format = Format.TEXT
    def dump(self, obj):
        return Vector._to_db(obj).encode('utf8')


class VectorBinaryDumper(VectorDumper):
    format = Format.BINARY
    def dump(self, obj):
        return Vector._to_db_binary(obj)


class VectorLoader(Loader):
    format = Format.TEXT
    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return Vector._from_db(data.decode('utf8'))


class VectorBinaryLoader(VectorLoader):
    format = Format.BINARY
    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return Vector._from_db_binary(data)


class VexDB(VectorDB):
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
        db_config: VexDBConfigDict,
        db_case_config: VexDBIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "VexDB"
        self.case_config = db_case_config
        self.table_name = db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._index_name = "vexdb_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"

        self.create_index_before_load = self.case_config.index_param()["create_index_before_load"]
        self.create_index_after_load = self.case_config.create_index_after_load

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        log.info(f"{self.name} config values: {self.connect_config}\n{self.case_config}")
        if not any(
            (
                self.create_index_before_load,
                self.create_index_after_load,
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
            if self.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None


    def __reduce__(self):
        return (
            self.__class__,
            (
                self.dim,
                {"connect_config": self.connect_config, "table_name": self.table_name}, 
                self.case_config,
            ),
        )

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)

        # register floatvector type
        floatvector_info = TypeInfo.fetch(conn, 'floatvector')
        if floatvector_info is None:
            raise psycopg.ProgrammingError('floatvector type not found in the database')
        floatvector_info.register(conn)

        # add oid to anonymous class for set_types
        text_dumper = type('', (VectorDumper,), {'oid': floatvector_info.oid})
        binary_dumper = type('', (VectorBinaryDumper,), {'oid': floatvector_info.oid})

        adapters = conn.adapters
        adapters.register_dumper('numpy.ndarray', text_dumper)
        adapters.register_dumper('numpy.ndarray', binary_dumper)
        adapters.register_dumper(Vector, text_dumper)
        adapters.register_dumper(Vector, binary_dumper)
        adapters.register_loader(floatvector_info.oid, VectorLoader)
        adapters.register_loader(floatvector_info.oid, VectorBinaryLoader)

        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _generate_search_query(self) -> sql.Composed:
        search_vector = sql.Placeholder()

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
                sql.SQL(" {search_vector}::floatvector({dim}) LIMIT %s::int").format(
                    search_vector=search_vector,
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

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.create_index_after_load:
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
                    sql.Identifier(self.connect_config["user"]),
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
                        ({primary_field} BIGINT PRIMARY KEY, embedding floatvector({dim}), {label_field} VARCHAR(64));
                        """,  # noqa: E501
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                        label_field=sql.Identifier(self._scalar_label_field),
                    )
                )
            else:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding floatvector({dim}));
                        """
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                    )
                )

            # PGVECTOR有，注释掉
            # self.cursor.execute(
            #     sql.SQL(
            #         "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
            #     ).format(table_name=sql.Identifier(self.table_name)),
            # )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create VexDB table: {self.table_name} error: {e}")
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
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT 'BINARY')").format(
                    table_name=sql.Identifier(self.table_name)
                )
            ) as copy:
                for i, row in enumerate(metadata_arr):
                    if self.with_scalar_labels:
                        copy.set_types(["bigint", "floatvector", "varchar"])
                        copy.write_row((row, embeddings_arr[i], labels_data[i]))
                    else:
                        copy.set_types(["bigint", "floatvector"])
                        copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into VexDB table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for VexDB - {filters}"
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
