"""Wrapper around the Hologres vector database over VectorDB"""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO
from typing import Any

import psycopg
from psycopg import Connection, Cursor, sql

from ..api import VectorDB
from .config import HologresConfig, HologresIndexConfig

log = logging.getLogger(__name__)


class Hologres(VectorDB):
    """Use psycopg instructions"""

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _tg_name: str = "vdb_bench_tg_1"

    _use_prepared_query: bool = False
    _prepared_query_q_k: str = "prepared_query_q_k"
    _prepared_query_q_f_k: str = "prepared_query_q_f_k"

    def __init__(
        self,
        dim: int,
        db_config: HologresConfig,
        db_case_config: HologresIndexConfig,
        collection_name: str = "vector_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Hologres"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # create vector extension
        if self.case_config.is_proxima():
            self.cursor.execute("CREATE EXTENSION proxima;")
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
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """

        self.conn, self.cursor = self._create_connection(**self.db_config)

        if self._use_prepared_query:
            self._prepare_query()

        self._set_search_guc()

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _prepare_query(self):
        self.cursor.execute(
            sql.SQL(
                """
                -- DEALLOCATE {prepared_query};
                PREPARE {prepared_query}(real[], int) AS
                SELECT
                    id
                FROM {table_name}
                ORDER BY {distance_function}(embedding, $1::REAL[]) {order_direction}
                LIMIT $2::INT
                """
            ).format(
                prepared_query=sql.SQL(self._prepared_query_q_k),
                distance_function=sql.SQL(self.case_config.distance_function()),
                table_name=sql.Identifier(self.table_name),
                order_direction=sql.SQL(self.case_config.order_direction()),
            )
        )
        self.cursor.execute(
            sql.SQL(
                """
                -- DEALLOCATE {prepared_query};
                PREPARE {prepared_query}(real[], bigint, int) AS
                SELECT
                    id
                FROM {table_name}
                WHERE id >= $2
                ORDER BY {distance_function}(embedding, $1::REAL[]) {order_direction}
                LIMIT $3::INT;
                """
            ).format(
                prepared_query=sql.SQL(self._prepared_query_q_f_k),
                distance_function=sql.SQL(self.case_config.distance_function()),
                table_name=sql.Identifier(self.table_name),
                order_direction=sql.SQL(self.case_config.order_direction()),
            )
        )

    def _set_search_guc(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        sql_guc = sql.SQL(f"SET hg_vector_ef_search = {self.case_config.ef_search};")
        log.info(f"{self.name} client set search guc: {sql_guc.as_string()}")
        self.cursor.execute(sql_guc)
        self.conn.commit()

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        log.info(f"{self.name} client drop table : {self.table_name}")
        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {table_name};").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

        try:
            log.info(f"{self.name} client drop table group : {self._tg_name}")
            self.cursor.execute(sql.SQL(f"CALL HG_DROP_TABLE_GROUP('{self._tg_name}');"))
        except Exception as e:
            log.info(f"{self.name} client drop table group : {self._tg_name} failed, error: {e}, ignore.")
        finally:
            self.conn.commit()

    def optimize(self, data_size: int | None = None):
        if self.case_config.create_index_after_load:
            self._create_index()
        self._full_compact()
        self._analyze()

    def _vacuum(self):
        log.info(f"{self.name} client vacuum table : {self.table_name}")
        try:
            # VACUUM cannot run inside a transaction block
            # it's better to new a connection
            self.conn.autocommit = True
            with self.conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        VACUUM {table_name};
                        """
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                    )
                )
            log.info(f"{self.name} client vacuum table : {self.table_name} done")
        except Exception as e:
            log.warning(f"Failed to vacuum table: {self.table_name} error: {e}")
            raise e from None
        finally:
            self.conn.autocommit = False

    def _analyze(self):
        log.info(f"{self.name} client analyze table : {self.table_name}")
        self.cursor.execute(sql.SQL(f"ANALYZE {self.table_name};"))
        log.info(f"{self.name} client analyze table : {self.table_name} done")

    def _full_compact(self):
        log.info(f"{self.name} client full compact table : {self.table_name}")
        self.cursor.execute(
            sql.SQL(
                """
                SELECT hologres.hg_full_compact_table(
                    '{table_name}',
                    'max_file_size_mb={full_compact_max_file_size_mb}'
                );
                """
            ).format(
                table_name=sql.SQL(self.table_name),
                full_compact_max_file_size_mb=sql.SQL(str(self.case_config.full_compact_max_file_size_mb)),
            )
        )
        log.info(f"{self.name} client full compact table : {self.table_name} done")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        sql_index = sql.SQL(
            """
            CALL set_table_property ('{table_name}', 'vectors', '{{
                "embedding": {{
                    "algorithm": "{algorithm}",
                    "distance_method": "{distance_method}",
                    "builder_params": {builder_params}
                }}
            }}');
            """
        ).format(
            table_name=sql.Identifier(self.table_name),
            algorithm=sql.SQL(self.case_config.algorithm()),
            distance_method=sql.SQL(self.case_config.distance_method()),
            builder_params=sql.SQL(json.dumps(self.case_config.builder_params())),
        )

        log.info(f"{self.name} client create index on table : {self.table_name}, with sql: {sql_index.as_string()}")
        try:
            self.cursor.execute(sql_index)
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create index on table: {self.table_name} error: {e}")
            raise e from None

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        sql_tg = sql.SQL(f"CALL HG_CREATE_TABLE_GROUP ('{self._tg_name}', 1);")
        log.info(f"{self.name} client create table group : {self._tg_name}, with sql: {sql_tg}")
        try:
            self.cursor.execute(sql_tg)
        except Exception as e:
            log.warning(f"Failed to create table group : {self._tg_name} error: {e}, ignore")
        finally:
            self.conn.commit()

        sql_table = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGINT PRIMARY KEY,
                embedding FLOAT4[] CHECK (array_ndims(embedding) = 1 AND array_length(embedding, 1) = {dim})
            )
            WITH (table_group = {tg_name});
            """
        ).format(
            table_name=sql.Identifier(self.table_name),
            dim=dim,
            tg_name=sql.SQL(self._tg_name),
        )
        log.info(f"{self.name} client create table : {self.table_name}, with sql: {sql_table.as_string()}")
        try:
            self.cursor.execute(sql_table)
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create table : {self.table_name} error: {e}")
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
            buffer = StringIO()
            for i in range(len(metadata)):
                buffer.write("%d\t%s\n" % (metadata[i], "{" + ",".join("%f" % x for x in embeddings[i]) + "}"))
            buffer.seek(0)

            with self.cursor.copy(
                sql.SQL("COPY {table_name} FROM STDIN").format(table_name=sql.Identifier(self.table_name))
            ) as copy:
                copy.write(buffer.getvalue())
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def _compose_query_and_params(self, vec: list[float], topk: int, ge_id: int | None = None):
        parts = []
        params = []

        where_clause = sql.SQL("")
        if ge_id is not None:
            where_clause = sql.SQL(" WHERE id >= %s ")
            params.append(ge_id)

        params.append('{' + ",".join(["%f" % x for x in vec]) + '}')
        params.append(topk)

        query = sql.SQL(
            """
            SELECT id
            FROM {table_name}
            {where_clause}
            ORDER BY {distance_function}(embedding, %s)
            {order_direction}
            LIMIT %s;
            """
        ).format(
            table_name=sql.Identifier(self.table_name),
            distance_function=sql.SQL(self.case_config.distance_function()),
            where_clause=where_clause,
            order_direction=sql.SQL(self.case_config.order_direction()),
        )

        return query, params

    def _search_prepared_query(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        q = "'{" + ",".join("%f" % i for i in query) + "}'"
        if filters:
            ge = filters.get("id")
            result = self.cursor.execute(sql.SQL(f"EXECUTE {self._prepared_query_q_f_k}({q}, {ge}, {k});"))
        else:
            result = self.cursor.execute(sql.SQL(f"EXECUTE {self._prepared_query_q_k}({q}, {k})"))
        return [int(i[0]) for i in result.fetchall()]

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if self._use_prepared_query:
            return self._search_prepared_query(query, k, filters, timeout)

        ge = filters.get("id") if filters else None
        q, params = self._compose_query_and_params(query, k, ge)
        result = self.cursor.execute(q, params, prepare=True)
        return [int(i[0]) for i in result.fetchall()]
