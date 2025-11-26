"""CockroachDB vector database client with connection pooling and retry logic."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql
from psycopg_pool import ConnectionPool

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import CockroachDBIndexConfig
from .db_retry import db_retry

log = logging.getLogger(__name__)


class CockroachDB(VectorDB):
    """
    CockroachDB vector database client:
    - Connection pooling (100+ connections for high throughput)
    - Automatic retry for serialization errors (40001, 40003)
    - Vector index support (C-SPANN algorithm)
    - Multi-region resilience
    """

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(  # noqa: PLR0915
        self,
        dim: int,
        db_config: dict,
        db_case_config: CockroachDBIndexConfig | None,
        collection_name: str = "vdbbench_cockroachdb",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "CockroachDB"
        self.case_config = db_case_config
        self.table_name = collection_name

        # Handle both dict-style config (from to_dict()) and direct dict
        if "connect_config" in db_config:
            self.connect_config = db_config["connect_config"]
            self.pool_size = db_config.get("pool_size", 100)
            self.max_overflow = db_config.get("max_overflow", 100)
            self.pool_recycle = db_config.get("pool_recycle", 3600)
        else:
            # Direct connection config for tests
            conn_params = {
                "host": db_config.get("host", "localhost"),
                "port": db_config.get("port", 26257),
                "dbname": db_config.get("db_name", "defaultdb"),
                "user": db_config.get("user_name", "root"),
                "password": db_config.get("password", ""),
            }
            # Add sslmode if specified, otherwise default to disable for local dev
            conn_params["sslmode"] = db_config.get("sslmode", "disable")

            self.connect_config = conn_params
            self.pool_size = db_config.get("pool_size", 100)
            self.max_overflow = db_config.get("max_overflow", 100)
            self.pool_recycle = db_config.get("pool_recycle", 3600)

        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._index_name = f"{self.table_name}_vector_idx"
        self._primary_field = "id"  # UUID for distribution
        self._metadata_field = "metadata_id"  # BIGINT for framework compatibility
        self._vector_field = "embedding"
        self._scalar_label_field = "label"

        self.pool: ConnectionPool | None = None
        self.conn: Connection | None = None
        self.cursor: Cursor | None = None

        log.info(f"{self.name} config: {self.connect_config}, pool_size={self.pool_size}")

        # Allow manual index creation (both flags can be False)
        # This is useful when CREATE INDEX times out in subprocess on multi-node clusters
        if self.case_config is not None and not any(
            (self.case_config.create_index_before_load, self.case_config.create_index_after_load)
        ):
            log.warning(f"{self.name}: Both create_index flags are False - expecting manually created index")

        # Initialize with temporary connection for setup
        conn, cursor = self._create_connection(**self.connect_config)
        try:
            # Enable pgvector extension (in transaction)
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            cursor.close()

            # Enable vector indexes at cluster level (requires autocommit, not in transaction)
            conn.autocommit = True
            cursor = conn.cursor()
            try:
                cursor.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
            except Exception as e:
                # May already be enabled or permission issue, log and continue
                log.warning(f"Could not enable vector indexes: {e}")
            cursor.close()

            # Reset to transaction mode for remaining operations
            conn.autocommit = False
            cursor = conn.cursor()

            if drop_old:
                if self.case_config is not None:
                    self._drop_index()  # Use SQLAlchemy
                self._drop_table(cursor, conn)
                self._create_table(cursor, conn, dim)
                if self.case_config is not None and self.case_config.create_index_before_load:
                    self._create_index()  # Use SQLAlchemy
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        """Create a single connection with pgvector support."""
        conn = psycopg.connect(**kwargs)
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()
        return conn, cursor

    def _create_connection_pool(self) -> ConnectionPool:
        """Create connection pool with production settings."""
        # Build connection info without 'options' parameter (not supported by psycopg_pool)
        conninfo = (
            f"host={self.connect_config['host']} "
            f"port={self.connect_config['port']} "
            f"dbname={self.connect_config['dbname']} "
            f"user={self.connect_config['user']} "
            f"password={self.connect_config['password']}"
        )

        # Add sslmode if present
        if "sslmode" in self.connect_config:
            conninfo += f" sslmode={self.connect_config['sslmode']}"

        # Add statement timeout for long-running vector index operations
        conninfo += " options='-c statement_timeout=600s'"

        # Configure each connection with vector support and search parameters
        def configure_connection(conn: Connection) -> None:
            register_vector(conn)
            # Set vector_search_beam_size on every connection for index usage
            if self.case_config is not None:
                search_param = self.case_config.search_param()
                beam_size = search_param.get("vector_search_beam_size", 32)
                with conn.cursor() as cur:
                    cur.execute(f"SET vector_search_beam_size = {beam_size}")
                conn.commit()

        return ConnectionPool(
            conninfo=conninfo,
            min_size=self.pool_size,
            max_size=self.pool_size + self.max_overflow,
            max_lifetime=self.pool_recycle,
            max_idle=300,
            reconnect_timeout=10.0,
            configure=configure_connection,
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Initialize connection pool for benchmark operations."""
        self.pool = self._create_connection_pool()

        try:
            with self.pool.connection() as conn:
                conn.autocommit = False
                self.conn = conn
                self.cursor = conn.cursor()

                # Set session parameters (only if case_config is provided)
                if self.case_config is not None:
                    session_options = self.case_config.session_param()["session_options"]
                    for setting in session_options:
                        param = setting["parameter"]
                        command = sql.SQL("SET {setting_name} = {val};").format(
                            setting_name=sql.Identifier(param["setting_name"]),
                            val=sql.Literal(int(param["val"])),
                        )
                        log.debug(command.as_string(self.cursor))
                        self.cursor.execute(command)
                    conn.commit()

                yield
        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            if self.pool:
                self.pool.close()
            self.cursor = None
            self.conn = None
            self.pool = None

    @db_retry(max_attempts=3, initial_delay=0.5, backoff_factor=2.0)
    def _drop_table(self, cursor: Cursor, conn: Connection):
        """Drop table with retry logic."""
        log.info(f"{self.name} dropping table: {self.table_name}")
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {table_name} CASCADE").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        conn.commit()

    def _drop_index(self):
        """Drop CockroachDB vector index if it exists (DDL with autocommit)."""
        log.info(f"{self.name} dropping index: {self._index_name}")
        conn = psycopg.connect(**self.connect_config)
        conn.autocommit = True
        try:
            cursor = conn.cursor()
            cursor.execute(f"DROP INDEX IF EXISTS {self._index_name}")
            cursor.close()
        finally:
            conn.close()

    @db_retry(max_attempts=3, initial_delay=0.5, backoff_factor=2.0)
    def _create_table(self, cursor: Cursor, conn: Connection, dim: int):
        """Create table with VECTOR column."""
        log.info(f"{self.name} creating table: {self.table_name}")

        # CockroachDB best practice: Use UUID primary key to avoid hotspots in distributed deployments
        # Keep metadata_id as BIGINT for framework compatibility
        if self.with_scalar_labels:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table_name}
                    ({primary_field} UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                     {metadata_field} BIGINT NOT NULL,
                     {vector_field} VECTOR({dim}),
                     {label_field} VARCHAR(64));
                    """,
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    metadata_field=sql.Identifier(self._metadata_field),
                    vector_field=sql.Identifier(self._vector_field),
                    label_field=sql.Identifier(self._scalar_label_field),
                    dim=dim,
                )
            )
        else:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table_name}
                    ({primary_field} UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                     {metadata_field} BIGINT NOT NULL,
                     {vector_field} VECTOR({dim}));
                    """
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    metadata_field=sql.Identifier(self._metadata_field),
                    vector_field=sql.Identifier(self._vector_field),
                    dim=dim,
                )
            )

        # Note: CockroachDB doesn't support SET STORAGE PLAIN (PostgreSQL-specific)
        # Vector columns are handled automatically
        conn.commit()

    def _create_index(self):
        """Create CockroachDB vector index (DDL with autocommit)."""
        log.info(f"{self.name} creating vector index: {self._index_name}")

        index_param = self.case_config.index_param()

        # Build WITH clause for index parameters
        options_list = []
        for option in index_param["index_creation_with_options"]:
            if option["val"] is not None:
                options_list.append(f"{option['option_name']} = {option['val']}")

        with_clause = f" WITH ({', '.join(options_list)})" if options_list else ""

        # Build SQL string (DDL - no need for parameterization)
        sql_str = (
            f"CREATE VECTOR INDEX IF NOT EXISTS {self._index_name} "
            f"ON {self.table_name} ({self._vector_field} {index_param['metric']})"
            f"{with_clause}"
        )

        log.info(f"Creating index with SQL: {sql_str}")

        # Use autocommit for DDL
        conn = psycopg.connect(**self.connect_config)
        conn.autocommit = True
        try:
            cursor = conn.cursor()
            cursor.execute(sql_str)
            cursor.close()
        finally:
            conn.close()

    def _wait_for_index_creation(self, start_time: float) -> None:
        """Wait for background index creation to complete after connection timeout."""
        import time

        max_wait = 300  # 5 minutes max
        poll_interval = 5
        waited = 0

        while waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval

            # Create fresh connection to check status
            try:
                check_conn = psycopg.connect(**self.connect_config)
                check_cursor = check_conn.cursor()
                try:
                    # Check if index exists
                    check_cursor.execute(
                        "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s",
                        (self.table_name, self._index_name),
                    )
                    if check_cursor.fetchone():
                        # Index exists! Verify it's usable by doing a quick test query
                        try:
                            from psycopg import sql

                            check_cursor.execute(
                                sql.SQL("SELECT 1 FROM {} LIMIT 1").format(sql.Identifier(self.table_name))
                            )
                            check_cursor.fetchone()
                            total_time = time.time() - start_time
                            log.info(f"Index {self._index_name} created successfully (total time: {total_time:.1f}s)")
                        except Exception as query_error:
                            # Index not yet usable
                            log.info(f"Index exists but not yet usable... ({waited}s elapsed, error: {query_error})")
                        else:
                            return
                finally:
                    check_cursor.close()
                    check_conn.close()
            except Exception as check_error:
                log.warning(f"Error checking index status: {check_error}")
                # Continue waiting

        # Timeout waiting for index
        msg = f"Timeout waiting for index {self._index_name} after {waited}s"
        log.error(msg)
        raise RuntimeError(msg)

    def optimize(self, data_size: int | None = None):
        """Post-insert optimization: create index if needed.

        Note: On multi-node clusters, CockroachDB v25.4 may close connections
        at 30s during CREATE VECTOR INDEX from subprocess contexts. The index
        creation continues in background. We handle this gracefully by checking
        if the index was created successfully after timeout.
        """
        log.info(f"{self.name} post-insert optimization")
        if self.case_config is not None and self.case_config.create_index_after_load:
            import time

            # Build CREATE INDEX SQL
            index_param = self.case_config.index_param()
            options_list = []
            for option in index_param["index_creation_with_options"]:
                if option["val"] is not None:
                    options_list.append(f"{option['option_name']} = {option['val']}")

            with_clause = f" WITH ({', '.join(options_list)})" if options_list else ""
            sql_str = (
                f"CREATE VECTOR INDEX IF NOT EXISTS {self._index_name} "
                f"ON {self.table_name} ({self._vector_field} {index_param['metric']})"
                f"{with_clause}"
            )

            log.info(f"{self.name} creating vector index: {self._index_name}")
            log.info(f"Index SQL: {sql_str}")

            start_time = time.time()
            connection_closed = False

            # Try to create index
            try:
                with self.pool.connection() as conn:
                    register_vector(conn)
                    conn.autocommit = True
                    cursor = conn.cursor()
                    try:
                        cursor.execute(sql_str)
                        elapsed = time.time() - start_time
                        log.info(f"{self.name} index created successfully in {elapsed:.1f}s")
                        return  # Success!
                    finally:
                        cursor.close()
            except Exception as e:
                elapsed = time.time() - start_time
                # Check if this is the expected 30s timeout on multi-node clusters
                if "server closed the connection" in str(e) or "connection" in str(e).lower():
                    log.warning(f"Connection closed after {elapsed:.1f}s during index creation: {e}")
                    log.info("This is expected on multi-node clusters - checking if index was created...")
                    connection_closed = True
                else:
                    # Unexpected error, re-raise
                    raise

            # Connection closed - wait for background index creation to complete
            if connection_closed:
                self._wait_for_index_creation(start_time)

    @db_retry(max_attempts=3, initial_delay=0.5, backoff_factor=2.0)
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings with COPY for performance."""
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if self.with_scalar_labels:
            assert labels_data is not None, "labels_data required when with_scalar_labels=True"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            # UUID primary key is auto-generated, we only insert metadata_id and embedding
            with self.cursor.copy(
                sql.SQL(
                    "COPY {table_name} ({metadata_field}, {vector_field}{label_field}) FROM STDIN (FORMAT BINARY)"
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    metadata_field=sql.Identifier(self._metadata_field),
                    vector_field=sql.Identifier(self._vector_field),
                    label_field=sql.SQL(f", {self._scalar_label_field}") if self.with_scalar_labels else sql.SQL(""),
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
            log.warning(f"Failed to insert data into {self.table_name}: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        """Prepare WHERE clause for filtered queries."""
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            # Filter on metadata_id, not UUID primary key
            self.where_clause = f"WHERE {self._metadata_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Unsupported filter for CockroachDB: {filters}"
            raise ValueError(msg)

    def ready_to_load(self) -> bool:
        """Check if ready to load data."""

    @db_retry(max_attempts=3, initial_delay=0.5, backoff_factor=2.0)
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        """Search for k nearest neighbors using vector index."""
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # Use default L2 distance if no case_config provided
        if self.case_config is not None:
            search_param = self.case_config.search_param()
            metric_op = search_param["metric_fun_op"]
        else:
            metric_op = "<->"  # Default to L2 distance

        q = np.asarray(query)

        # Build search query - return metadata_id for framework compatibility
        search_sql = sql.SQL("SELECT {metadata_field} FROM {table_name} {where_clause} ORDER BY {vector_field}").format(
            metadata_field=sql.Identifier(self._metadata_field),
            table_name=sql.Identifier(self.table_name),
            where_clause=sql.SQL(getattr(self, "where_clause", "")),
            vector_field=sql.Identifier(self._vector_field),
        )

        # Add distance operator and limit
        full_sql = search_sql + sql.SQL(" {metric_op} %s LIMIT %s").format(
            metric_op=sql.SQL(metric_op),
        )

        result = self.cursor.execute(full_sql, (q, k), prepare=True, binary=True)
        return [int(i[0]) for i in result.fetchall()]
