"""Wrapper around the Hologres vector database over VectorDB"""

import json
import logging
import struct
import threading
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO
from typing import Any

import psycopg
from psycopg import Connection, Cursor, sql
from psycopg.adapt import Dumper
from psycopg.pq import Format

from ..api import VectorDB
from .config import HologresConfig, HologresIndexConfig

log = logging.getLogger(__name__)


class HoloFloat4Array:
    """Lightweight wrapper for float arrays - 1 object per query instead of 768 Float4 wrappers."""

    __slots__ = ("data",)

    def __init__(self, data: list[float]):
        self.data = data


class HoloFloat4ArrayDumper(Dumper):
    """Custom Dumper that serializes float arrays to PostgreSQL binary float4[] format."""

    format = Format.BINARY
    oid = 1021  # PostgreSQL OID for float4[] (_float4)

    def dump(self, obj: HoloFloat4Array) -> bytes:
        # PostgreSQL binary array format:
        # Header: ndim(4) + has_null(4) + OID elemtype(4) + dim_size(4) + lower_bound(4) = 20 bytes
        # For each element: itemlen(4) + float32(4) = 8 bytes per element
        header = struct.pack(">iiIii", 1, 0, 700, len(obj.data), 1)
        return header + b"".join(struct.pack(">if", 4, x) for x in obj.data)


# Register HoloFloat4ArrayDumper globally so it's available for all connections and cursors.
# This must happen at module load time, before any connections/cursors are created,
# because psycopg cursors snapshot the adapter map at creation time.
psycopg.adapters.register_dumper(HoloFloat4Array, HoloFloat4ArrayDumper)


class Hologres(VectorDB):
    """Use psycopg instructions"""

    thread_safe: bool = True  # each thread gets its own connection via threading.local()

    _tg_name: str = "vdb_bench_tg_1"

    def __init__(
        self,
        dim: int,
        db_config: HologresConfig,
        db_case_config: HologresIndexConfig,
        collection_name: str = "vector_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Alibaba Cloud Hologres"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._primary_field = "id"
        self._vector_field = "embedding"

        # Thread-local storage for per-thread connections (thread-safe concurrent inserts)
        self._local = threading.local()
        self._all_conns: list[psycopg.Connection[Any]] = []
        self._conns_lock = threading.Lock()

        # Temporary connection for setup, closed at end of __init__
        conn, cursor = self._create_connection(**self.db_config)
        self._local.conn = conn
        self._local.cursor = cursor

        # create vector extension
        if self.case_config.is_proxima():
            cursor.execute("CREATE EXTENSION proxima;")
            conn.commit()

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
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        cursor.close()
        conn.close()
        self._local.cursor = None
        self._local.conn = None

    def __getstate__(self):
        # Exclude unpicklable threading objects; recreated in __setstate__
        state = self.__dict__.copy()
        state.pop("_local", None)
        state.pop("_conns_lock", None)
        state.pop("_all_conns", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._all_conns = []

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        conn.autocommit = True
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _get_conn(self) -> Connection:
        """Return this thread's connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn, cursor = self._create_connection(**self.db_config)
            self._local.conn = conn
            self._local.cursor = cursor
            self._set_search_guc_on(conn, cursor)
            with self._conns_lock:
                self._all_conns.append(conn)
        return conn

    def _get_cursor(self) -> Cursor:
        """Return this thread's cursor, creating connection if needed."""
        cursor = getattr(self._local, "cursor", None)
        if cursor is None or cursor.closed:
            self._get_conn()  # creates both conn and cursor
            cursor = self._local.cursor
        return cursor

    def _set_search_guc_on(self, conn: Connection, cursor: Cursor) -> None:
        """Set session-level search GUC on the given connection."""
        sql_guc = sql.SQL(f"SET hg_vector_ef_search = {self.case_config.ef_search};")
        log.info(f"{self.name} client set search guc: {sql_guc.as_string()}")
        cursor.execute(sql_guc)
        conn.commit()

    def _set_search_guc(self) -> None:
        """Set session-level search GUC on this thread's connection."""
        self._set_search_guc_on(self._get_conn(), self._get_cursor())

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Open connection for calling thread, prepare for operations.

        Worker threads lazily create their own connections via _get_conn().
        On exit, closes all thread-local connections.
        """
        conn, cursor = self._create_connection(**self.db_config)
        self._local.conn = conn
        self._local.cursor = cursor
        self._set_search_guc_on(conn, cursor)
        with self._conns_lock:
            self._all_conns.append(conn)

        self._search_query_no_filter = sql.SQL("""
            SELECT id
            FROM {table_name}
            ORDER BY {distance_function}(embedding, %b)
            {order_direction}
            LIMIT %s;
            """).format(
            table_name=sql.Identifier(self.table_name),
            distance_function=sql.SQL(self.case_config.distance_function()),
            order_direction=sql.SQL(self.case_config.order_direction()),
        )

        self._search_query_with_filter = sql.SQL("""
            SELECT id
            FROM {table_name}
            WHERE id >= %s
            ORDER BY {distance_function}(embedding, %b)
            {order_direction}
            LIMIT %s;
            """).format(
            table_name=sql.Identifier(self.table_name),
            distance_function=sql.SQL(self.case_config.distance_function()),
            order_direction=sql.SQL(self.case_config.order_direction()),
        )

        try:
            yield
        finally:
            with self._conns_lock:
                conns_to_close = list(self._all_conns)
                self._all_conns.clear()
            for c in conns_to_close:
                if not c.closed:
                    c.close()
            self._local.conn = None
            self._local.cursor = None

    def _drop_table(self):
        conn = self._get_conn()
        cursor = self._get_cursor()

        log.info(f"{self.name} client drop table : {self.table_name}")
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {table_name};").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        conn.commit()

        try:
            log.info(f"{self.name} client purge table recycle bin: {self.table_name}")
            cursor.execute(
                sql.SQL("purge TABLE {table_name};").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            )
        except Exception as e:
            log.info(f"{self.name} client purge table {self.table_name} recycle bin failed, error: {e}, ignore.")
        finally:
            conn.commit()

        try:
            log.info(f"{self.name} client drop table group : {self._tg_name}")
            cursor.execute(sql.SQL(f"CALL HG_DROP_TABLE_GROUP('{self._tg_name}');"))
        except Exception as e:
            log.info(f"{self.name} client drop table group : {self._tg_name} failed, error: {e}, ignore.")
        finally:
            conn.commit()

        try:
            log.info(f"{self.name} client free cache")
            cursor.execute("select hg_admin_command('freecache');")
        except Exception as e:
            log.info(f"{self.name} client free cache failed, error: {e}, ignore.")
        finally:
            conn.commit()

    def optimize(self, data_size: int | None = None):
        if self.case_config.create_index_after_load:
            self._create_index()
        self._full_compact()
        self._analyze()

    def _vacuum(self):
        conn = self._get_conn()
        log.info(f"{self.name} client vacuum table : {self.table_name}")
        try:
            # VACUUM cannot run inside a transaction block
            # it's better to new a connection
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("""
                        VACUUM {table_name};
                        """).format(
                        table_name=sql.Identifier(self.table_name),
                    )
                )
            log.info(f"{self.name} client vacuum table : {self.table_name} done")
        except Exception as e:
            log.warning(f"Failed to vacuum table: {self.table_name} error: {e}")
            raise e from None
        finally:
            conn.autocommit = True

    def _analyze(self):
        cursor = self._get_cursor()
        log.info(f"{self.name} client analyze table : {self.table_name}")
        cursor.execute(sql.SQL(f"ANALYZE {self.table_name};"))
        log.info(f"{self.name} client analyze table : {self.table_name} done")

    def _full_compact(self):
        cursor = self._get_cursor()
        log.info(f"{self.name} client full compact table : {self.table_name}")
        cursor.execute(
            sql.SQL("""
                SELECT hologres.hg_full_compact_table(
                    '{table_name}',
                    'max_file_size_mb={full_compact_max_file_size_mb}'
                );
                """).format(
                table_name=sql.SQL(self.table_name),
                full_compact_max_file_size_mb=sql.SQL(str(self.case_config.full_compact_max_file_size_mb)),
            )
        )
        log.info(f"{self.name} client full compact table : {self.table_name} done")

    def _create_index(self):
        conn = self._get_conn()
        cursor = self._get_cursor()

        sql_index = sql.SQL("""
            CALL set_table_property ('{table_name}', 'vectors', '{{
                "embedding": {{
                    "algorithm": "{algorithm}",
                    "distance_method": "{distance_method}",
                    "builder_params": {builder_params}
                }}
            }}');
            """).format(
            table_name=sql.Identifier(self.table_name),
            algorithm=sql.SQL(self.case_config.algorithm()),
            distance_method=sql.SQL(self.case_config.distance_method()),
            builder_params=sql.SQL(json.dumps(self.case_config.builder_params())),
        )

        log.info(f"{self.name} client create index on table : {self.table_name}, with sql: {sql_index.as_string()}")
        try:
            cursor.execute(sql_index)
            conn.commit()
        except Exception as e:
            log.warning(f"Failed to create index on table: {self.table_name} error: {e}")
            raise e from None

    def _set_replica_count(self, replica_count: int = 2):
        conn = self._get_conn()
        cursor = self._get_cursor()

        try:
            # non-warehouse mode by default
            sql_tg_replica = sql.SQL(
                f"CALL hg_set_table_group_property('{self._tg_name}', 'replica_count', '{replica_count}');"
            )

            # check warehouse mode
            sql_check = sql.SQL("select count(*) from hologres.hg_warehouses;")
            log.info(f"check warehouse mode with sql: {sql_check}")
            cursor.execute(sql_check)
            result_check = cursor.fetchone()[0]
            if result_check > 0:
                # get warehouse name
                sql_get_warehouse_name = sql.SQL("select current_warehouse();")
                log.info(f"get warehouse name with sql: {sql_get_warehouse_name}")
                cursor.execute(sql_get_warehouse_name)
                sql_tg_replica = sql.SQL("""
                    CALL hg_table_group_set_warehouse_replica_count (
                        '{dbname}.{tg_name}',
                        {replica_count},
                        '{warehouse_name}'
                    );
                    """).format(
                    tg_name=sql.SQL(self._tg_name),
                    warehouse_name=sql.SQL(cursor.fetchone()[0]),
                    dbname=sql.SQL(self.db_config["dbname"]),
                    replica_count=replica_count,
                )
            log.info(f"{self.name} client set table group replica: {self._tg_name}, with sql: {sql_tg_replica}")
            cursor.execute(sql_tg_replica)
        except Exception as e:
            log.warning(f"Failed to set replica count, error: {e}, ignore")
        finally:
            conn.commit()

    def _create_table(self, dim: int):
        conn = self._get_conn()
        cursor = self._get_cursor()

        sql_tg = sql.SQL(f"CALL HG_CREATE_TABLE_GROUP ('{self._tg_name}', 1);")
        log.info(f"{self.name} client create table group : {self._tg_name}, with sql: {sql_tg}")
        try:
            cursor.execute(sql_tg)
        except Exception as e:
            log.warning(f"Failed to create table group : {self._tg_name} error: {e}, ignore")
        finally:
            conn.commit()

        self._set_replica_count(replica_count=2)

        sql_table = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGINT PRIMARY KEY,
                embedding FLOAT4[] CHECK (array_ndims(embedding) = 1 AND array_length(embedding, 1) = {dim})
            )
            WITH (table_group = {tg_name});
            """).format(
            table_name=sql.Identifier(self.table_name),
            dim=dim,
            tg_name=sql.SQL(self._tg_name),
        )
        log.info(f"{self.name} client create table : {self.table_name}, with sql: {sql_table.as_string()}")
        try:
            cursor.execute(sql_table)
            conn.commit()
        except Exception as e:
            log.warning(f"Failed to create table : {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        conn = self._get_conn()
        cursor = self._get_cursor()

        try:
            buffer = StringIO()
            for i in range(len(metadata)):
                buffer.write("%d\t%s\n" % (metadata[i], "{" + ",".join("%f" % x for x in embeddings[i]) + "}"))
            buffer.seek(0)

            with cursor.copy(
                sql.SQL("COPY {table_name} FROM STDIN").format(table_name=sql.Identifier(self.table_name))
            ) as copy:
                copy.write(buffer.getvalue())
            conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        cursor = self._get_cursor()

        ge = filters.get("id") if filters else None
        q = HoloFloat4Array(query)

        if ge is not None:
            params = (ge, q, k)
            result = cursor.execute(self._search_query_with_filter, params, prepare=True, binary=True)
        else:
            params = (q, k)
            result = cursor.execute(self._search_query_no_filter, params, prepare=True, binary=True)

        return [int(i[0]) for i in result.fetchall()]
