import logging
import os
import struct
import time
from contextlib import contextmanager
from typing import Any

from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

from ..api import VectorDB
from .config import YDBIndexConfig

log = logging.getLogger(__name__)

YDB_USER_ENV = "YDB_USER"
YDB_PASSWORD_ENV = "YDB_PASSWORD"
YDB_SSL_ROOT_CERTIFICATES_FILE_ENV = "YDB_SSL_ROOT_CERTIFICATES_FILE"
YDB_CREDENTIAL_ENV_KEYS = (
    "YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS",
    YDB_USER_ENV,
    "YDB_ACCESS_TOKEN_CREDENTIALS",
    "YDB_OAUTH2_KEY_FILE",
)

YDB_LABEL_FIELD = "labels"
YDB_INDEX_WAIT_POLL_SECONDS = 5
YDB_INDEX_WAIT_TIMEOUT_SECONDS = 7200
YDB_DRIVER_WAIT_SECONDS = 30
YDB_DEFAULT_TABLE_PARTITION_SIZE_MB = 1000
YDB_DEFAULT_INDEX_PARTITION_SIZE_MB = 1000
YDB_DEFAULT_OPERATION_TIMEOUT_SECONDS = 24 * 3600
YDB_INDEX_IMPL_LEVEL_TABLE = "indexImplLevelTable"
YDB_INDEX_IMPL_POSTING_TABLE = "indexImplPostingTable"
YDB_INDEX_IMPL_PREFIX_TABLE = "indexImplPrefixTable"
YDB_VECTOR_INDEX_NAME = "vector_idx"
YDB_TRANSIENT_OP_MAX_ATTEMPTS = 5
YDB_TRANSIENT_OP_BACKOFF_SECONDS = 5


def convert_vector_to_bytes(vector: list[float]) -> bytes:
    values = [float(v) for v in vector]
    packed = struct.pack(f"<{len(values)}f", *values)
    return packed + b"\x01"


class YDB(VectorDB):
    """YDB vector search client using vector_kmeans_tree indexes."""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]
    thread_safe = True
    serial_search_in_process = True
    case_unique_collection_name = True
    case_filters_at_init = True
    optimize_via_picklable_worker = True

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: YDBIndexConfig,
        collection_name: str = "vdbbench_ydb",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        filters: Filter = non_filter,
        **kwargs,
    ):
        self.name = "YDB"
        self.db_config = db_config
        self.case_config = db_case_config
        table_from_config = db_config.get("table_name") or ""
        self.table_name = table_from_config or collection_name
        self.index_name = YDB_VECTOR_INDEX_NAME
        self.dim = dim
        self.filters = filters
        self.with_scalar_labels = with_scalar_labels or filters.type == FilterOp.StrEqual
        self._where_clause = ""
        self._label_filter_value: str | None = None
        self._index_ready = False

        self.driver = None
        self.pool = None

        if drop_old:
            with self._session_pool() as pool:
                self._drop_table(pool)
                self._create_table(pool)

    def __getstate__(self) -> dict[str, Any]:
        # YDB driver/session pool hold gRPC/protobuf objects that cannot be pickled.
        # ProcessPoolExecutor(spawn) sends the DB wrapper to load/optimize subprocesses.
        state = self.__dict__.copy()
        state["driver"] = None
        state["pool"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.driver = None
        self.pool = None

    @staticmethod
    def _resolve_login(db_config: dict) -> tuple[str, str]:
        user = db_config.get("user") or os.environ.get(YDB_USER_ENV, "")
        password = db_config.get("password") or os.environ.get(YDB_PASSWORD_ENV, "")
        return user, password

    @staticmethod
    def _has_sdk_credentials_env() -> bool:
        if any(os.environ.get(key) for key in YDB_CREDENTIAL_ENV_KEYS):
            return True
        if os.environ.get("YDB_ANONYMOUS_CREDENTIALS", "0") == "1":
            return True
        return os.environ.get("YDB_METADATA_CREDENTIALS", "0") == "1"

    @staticmethod
    def _ssl_root_certificates_path(db_config: dict) -> str | None:
        path = db_config.get("ssl_root_certificates_file") or os.environ.get(YDB_SSL_ROOT_CERTIFICATES_FILE_ENV)
        return path or None

    @staticmethod
    def _load_root_certificates(db_config: dict) -> bytes | None:
        import ydb

        path = YDB._ssl_root_certificates_path(db_config)
        if not path:
            return None
        root_certificates = ydb.load_ydb_root_certificate(path)
        if root_certificates is None:
            log.warning("YDB SSL root certificate file not found: %s", path)
            return None
        log.debug("Using YDB SSL root certificate from %s", path)
        return root_certificates

    @staticmethod
    def _driver_config(db_config: dict, credentials=None):
        import ydb

        kwargs: dict[str, Any] = {
            "endpoint": db_config["endpoint"],
            "database": db_config.get("database"),
        }
        root_certificates = YDB._load_root_certificates(db_config)
        if root_certificates is not None:
            kwargs["root_certificates"] = root_certificates
        if credentials is not None:
            kwargs["credentials"] = credentials
        return ydb.DriverConfig(**kwargs)

    @staticmethod
    def _build_credentials(db_config: dict):
        import ydb

        auth_mode = db_config.get("auth_mode", "env")
        if auth_mode == "anonymous":
            return ydb.AnonymousCredentials()
        if auth_mode == "token":
            token = db_config.get("token") or os.environ.get("YDB_ACCESS_TOKEN_CREDENTIALS", "")
            if not token:
                msg = "auth_mode=token requires a non-empty token"
                raise ValueError(msg)
            return ydb.AccessTokenCredentials(token)

        user, password = YDB._resolve_login(db_config)
        if auth_mode == "login" or user:
            if not user:
                msg = f"auth_mode=login requires --user or ${YDB_USER_ENV}"
                raise ValueError(msg)
            driver_config = YDB._driver_config(db_config)
            return ydb.StaticCredentials(driver_config, user, password)

        if YDB._has_sdk_credentials_env():
            return ydb.credentials_from_env_variables()

        log.debug("No YDB credentials in env; using anonymous auth for local server")
        return ydb.AnonymousCredentials()

    @staticmethod
    def _operation_timeout_seconds(db_config: dict) -> int:
        return int(db_config.get("operation_timeout_seconds", YDB_DEFAULT_OPERATION_TIMEOUT_SECONDS))

    @classmethod
    def _operation_settings(cls, db_config: dict):
        import ydb

        timeout = cls._operation_timeout_seconds(db_config)
        return (
            ydb.BaseRequestSettings()
            .with_timeout(timeout)
            .with_operation_timeout(timeout)
            .with_cancel_after(timeout)
        )

    @staticmethod
    def _wait_for_driver(driver, *, context: str) -> None:
        try:
            driver.wait(timeout=YDB_DRIVER_WAIT_SECONDS, fail_fast=True)
        except TimeoutError as exc:
            msg = f"YDB driver failed to connect within {YDB_DRIVER_WAIT_SECONDS}s ({context})"
            raise TimeoutError(msg) from exc

    @contextmanager
    def _session_pool(self):
        import ydb

        credentials = self._build_credentials(self.db_config)
        driver = ydb.Driver(driver_config=self._driver_config(self.db_config, credentials))
        pool = None
        try:
            self._wait_for_driver(driver, context="session pool")
            pool = ydb.QuerySessionPool(driver)
            yield pool
        finally:
            if pool is not None:
                pool.stop()
            driver.stop()

    @contextmanager
    def init(self):
        self._open_connection()
        try:
            yield
        finally:
            self._close_connection()

    def _open_connection(self) -> None:
        import ydb

        if self.pool is not None:
            return

        credentials = self._build_credentials(self.db_config)
        self.driver = ydb.Driver(driver_config=self._driver_config(self.db_config, credentials))
        self._wait_for_driver(self.driver, context="client init")
        self.pool = ydb.QuerySessionPool(self.driver)

    def _close_connection(self) -> None:
        if self.pool is not None:
            try:
                self.pool.stop()
            except Exception as exc:
                log.debug("Error stopping YDB session pool: %s", exc)
            self.pool = None
        if self.driver is not None:
            try:
                self.driver.stop()
            except Exception as exc:
                log.debug("Error stopping YDB driver: %s", exc)
            self.driver = None

    def _reconnect(self) -> None:
        self._close_connection()
        self._open_connection()

    def _drop_table(self, pool) -> None:
        pool.execute_with_retries(
            f"DROP TABLE IF EXISTS `{self.table_name}`",
            settings=self._operation_settings(self.db_config),
        )
        log.info("Dropped table %s", self.table_name)

    def _partition_count_settings_sql(self) -> str:
        min_count = self.db_config.get("auto_partitioning_min_partitions_count", 1000)
        max_count = self.db_config.get("auto_partitioning_max_partitions_count", 1100)
        return (
            f"AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = {min_count},\n"
            f"                AUTO_PARTITIONING_MAX_PARTITIONS_COUNT = {max_count}"
        )

    def _table_partitioning_settings_sql(self) -> str:
        partition_size_mb = self.db_config.get(
            "auto_partitioning_table_partition_size_mb",
            YDB_DEFAULT_TABLE_PARTITION_SIZE_MB,
        )
        return (
            "AUTO_PARTITIONING_BY_SIZE = ENABLED,\n"
            "                AUTO_PARTITIONING_BY_LOAD = ENABLED,\n"
            f"                AUTO_PARTITIONING_PARTITION_SIZE_MB = {partition_size_mb},\n"
            f"                {self._partition_count_settings_sql()}"
        )

    def _index_partitioning_settings_sql(self) -> str:
        partition_size_mb = self.db_config.get(
            "auto_partitioning_index_partition_size_mb",
            YDB_DEFAULT_INDEX_PARTITION_SIZE_MB,
        )
        return (
            "AUTO_PARTITIONING_BY_SIZE = ENABLED,\n"
            "                AUTO_PARTITIONING_BY_LOAD = ENABLED,\n"
            f"                AUTO_PARTITIONING_PARTITION_SIZE_MB = {partition_size_mb},\n"
            f"                {self._partition_count_settings_sql()}"
        )

    def _create_table_with_clause(self) -> str:
        return f"""
            WITH (
                {self._table_partitioning_settings_sql()}
            )"""

    def _index_impl_table_paths(self) -> list[str]:
        """Relative paths to vector index internal tables (see YDB vector index docs)."""
        base = f"{self.table_name}/{self.index_name}"
        paths = [f"{base}/{YDB_INDEX_IMPL_LEVEL_TABLE}"]
        if self.case_config.cover_embedding:
            paths.append(f"{base}/{YDB_INDEX_IMPL_POSTING_TABLE}")
        if len(self._resolved_index_on_columns()) > 1:
            paths.append(f"{base}/{YDB_INDEX_IMPL_PREFIX_TABLE}")
        return paths

    def _configure_index_table_partitioning(self, pool) -> None:
        settings_sql = self._index_partitioning_settings_sql()
        for table_path in self._index_impl_table_paths():
            for attempt in range(1, YDB_TRANSIENT_OP_MAX_ATTEMPTS + 1):
                try:
                    self.pool.execute_with_retries(
                        f"""
                        ALTER TABLE `{table_path}`
                        SET (
                            {settings_sql}
                        );
                        """,
                        settings=self._operation_settings(self.db_config),
                    )
                    log.info("Configured auto partitioning for YDB index table %s", table_path)
                    break
                except Exception as exc:
                    if self._is_transient_ydb_error(exc) and attempt < YDB_TRANSIENT_OP_MAX_ATTEMPTS:
                        log.warning(
                            "Transient YDB error configuring index table %s (%s/%s): %s",
                            table_path,
                            attempt,
                            YDB_TRANSIENT_OP_MAX_ATTEMPTS,
                            exc,
                        )
                        time.sleep(YDB_TRANSIENT_OP_BACKOFF_SECONDS)
                        self._reconnect()
                        continue
                    raise

    def _create_table(self, pool) -> None:
        label_column = f",\n                {YDB_LABEL_FIELD} Utf8" if self.with_scalar_labels else ""
        with_clause = self._create_table_with_clause()
        pool.execute_with_retries(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id Uint64 NOT NULL,
                embedding String NOT NULL{label_column},
                PRIMARY KEY (id)
            ){with_clause};
            """,
            settings=self._operation_settings(self.db_config),
        )
        log.info(
            "Created table %s (with_scalar_labels=%s, auto_partitioning_min=%s, auto_partitioning_max=%s)",
            self.table_name,
            self.with_scalar_labels,
            self.db_config.get("auto_partitioning_min_partitions_count", 1000),
            self.db_config.get("auto_partitioning_max_partitions_count", 1100),
        )

    def _resolved_index_on_columns(self) -> tuple[str, ...]:
        return self.case_config.index_on_columns(self.filters, with_scalar_labels=self.with_scalar_labels)

    def _index_on_sql(self) -> str:
        return ", ".join(self._resolved_index_on_columns())

    def _index_names_to_drop(self) -> tuple[str, ...]:
        legacy_index_name = f"{self.table_name}_vector_idx"
        return tuple(
            dict.fromkeys(
                (
                    self.index_name,
                    f"{self.index_name}__temp",
                    legacy_index_name,
                    f"{legacy_index_name}__temp",
                ),
            ),
        )

    def _drop_index_if_exists(self, pool, index_name: str) -> None:
        try:
            pool.execute_with_retries(
                f"ALTER TABLE `{self.table_name}` DROP INDEX `{index_name}`;",
                settings=self._operation_settings(self.db_config),
            )
            log.info("Dropped YDB vector index %s on %s", index_name, self.table_name)
        except Exception as exc:
            log.debug("Skip dropping YDB index %s on %s: %s", index_name, self.table_name, exc)

    def _drop_vector_indexes(self, pool) -> None:
        """Drop final/temp vector indexes so rebuilds do not hit stale scheme paths."""
        for index_name in self._index_names_to_drop():
            self._drop_index_if_exists(pool, index_name)

    @staticmethod
    def _is_index_path_exists_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "path exist" in message or "already exists" in message

    @staticmethod
    def _is_transient_ydb_error(exc: Exception) -> bool:
        message = str(exc).lower()
        exc_name = type(exc).__name__.lower()
        markers = (
            "unavailable",
            "connection to tablet was lost",
            "connection reset",
            "connection refused",
            "transport",
            "deadline exceeded",
            "socket",
            "broken pipe",
            "server overload",
            "cluster discovery",
            "400050",
        )
        if any(marker in message for marker in markers):
            return True
        return "unavailable" in exc_name or "timeout" in exc_name

    @classmethod
    def _is_index_build_in_progress_error(cls, exc: Exception) -> bool:
        return cls._is_index_path_exists_error(exc) or cls._is_transient_ydb_error(exc)

    def _ddl_retry_settings(self):
        import ydb

        # ADD INDEX must not be retried internally: a successful first attempt creates
        # the scheme path immediately, and a transparent retry fails with "path exist".
        return ydb.RetrySettings(max_retries=0)

    def _build_vector_index_ddl(self, index_name: str) -> str:
        index_param = self.case_config.index_param(self.filters, with_scalar_labels=self.with_scalar_labels)
        strategy = index_param["strategy"]
        on_sql = self._index_on_sql()
        cover_clause = index_param.get("cover_clause", "")
        cover_sql = f" {cover_clause}" if cover_clause else ""

        kmeans_options: list[str] = []
        for key in ("levels", "clusters", "overlap_clusters"):
            value = index_param.get(key)
            if value is not None:
                kmeans_options.append(f"{key}={value}")
        kmeans_options_sql = (
            ",\n                " + ",\n                ".join(kmeans_options) if kmeans_options else ""
        )

        return f"""
            ALTER TABLE `{self.table_name}`
            ADD INDEX {index_name}
            GLOBAL USING vector_kmeans_tree
            ON ({on_sql}){cover_sql}
            WITH (
                {strategy},
                vector_type="float",
                vector_dimension={self.dim}{kmeans_options_sql}
            );
            """

    def _create_vector_index(self, pool, index_name: str) -> None:
        ddl_settings = self._operation_settings(self.db_config)
        pool.execute_with_retries(
            self._build_vector_index_ddl(index_name),
            settings=ddl_settings,
            retry_settings=self._ddl_retry_settings(),
        )

    def _index_probe_query(self) -> tuple[str, dict]:
        import ydb

        search_param = self.case_config.search_param()
        knn_function = search_param["knn_function"]
        sort_order = search_param["sort_order"]
        probe_vector = convert_vector_to_bytes([0.0] * self.dim)
        query = f"""
        PRAGMA ydb.KMeansTreeSearchTopSize = "1";
        DECLARE $embedding AS String;

        SELECT id
        FROM `{self.table_name}` VIEW {self.index_name}
        ORDER BY Knn::{knn_function}(embedding, $embedding) {sort_order}
        LIMIT 1;
        """
        params = {"$embedding": (probe_vector, ydb.PrimitiveType.String)}
        return query, params

    def _probe_index_status(self, pool) -> str:
        query, params = self._index_probe_query()
        try:
            pool.execute_with_retries(query, params)
            return "ready"
        except Exception as exc:
            if self._is_transient_ydb_error(exc):
                return "connection_error"
            return "building"

    def _try_index_search(self, pool) -> bool:
        return self._probe_index_status(pool) == "ready"

    def _add_vector_index(self, pool) -> None:
        if self._try_index_search(pool):
            self._index_ready = True
            log.info(
                "Vector index %s is already searchable on %s",
                self.index_name,
                self.table_name,
            )
            return

        index_param = self.case_config.index_param(self.filters, with_scalar_labels=self.with_scalar_labels)
        on_sql = self._index_on_sql()
        cover_clause = index_param.get("cover_clause", "")
        cover_sql = f" {cover_clause}" if cover_clause else ""
        kmeans_options: list[str] = []
        for key in ("levels", "clusters", "overlap_clusters"):
            value = index_param.get(key)
            if value is not None:
                kmeans_options.append(f"{key}={value}")

        self._drop_vector_indexes(pool)
        try:
            self._create_vector_index(pool, self.index_name)
            log.info(
                "Created vector index %s on %s ON (%s)%s (kmeans_tree options: %s)",
                self.index_name,
                self.table_name,
                on_sql,
                cover_sql,
                ", ".join(kmeans_options) if kmeans_options else "server defaults",
            )
        except Exception as exc:
            if self._is_index_build_in_progress_error(exc):
                log.warning(
                    "YDB index build may still be in progress for %s on %s; waiting instead of failing: %s",
                    self.index_name,
                    self.table_name,
                    exc,
                )
                if self._is_transient_ydb_error(exc):
                    self._reconnect()
                return
            raise

    def _wait_for_index(self, pool) -> None:
        if self._index_ready:
            return

        deadline = time.monotonic() + YDB_INDEX_WAIT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            active_pool = getattr(self, "pool", None) or pool
            status = self._probe_index_status(active_pool)
            if status == "ready":
                self._index_ready = True
                log.info("Vector index %s is ready", self.index_name)
                return
            if status == "connection_error":
                log.warning(
                    "Lost YDB connection while waiting for vector index %s; reconnecting",
                    self.index_name,
                )
                self._reconnect()
            else:
                log.info("Waiting for vector index %s to become searchable", self.index_name)
            time.sleep(YDB_INDEX_WAIT_POLL_SECONDS)

        msg = f"Timed out waiting for YDB vector index {self.index_name} after {YDB_INDEX_WAIT_TIMEOUT_SECONDS}s"
        raise TimeoutError(msg)

    def optimize(self, data_size: int | None = None) -> None:
        if not self.case_config.create_index_after_load:
            log.info("Skipping vector index build (create_index_after_load=False)")
            return

        log.info(
            "Building YDB vector index for %d rows: levels=%s, clusters=%s, overlap_clusters=%s",
            data_size or 0,
            self.case_config.level,
            self.case_config.nlist,
            self.case_config.overlap_clusters,
        )
        self._add_vector_index(self.pool)
        self._wait_for_index(self.pool)
        self._configure_index_table_partitioning(self.pool)

    def prepare_filter(self, filters: Filter) -> None:
        self._label_filter_value = None
        if filters.type == FilterOp.NonFilter:
            self._where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self._where_clause = f"WHERE id >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self._where_clause = f"WHERE {YDB_LABEL_FIELD} = $label"
            self._label_filter_value = filters.label_value
        else:
            msg = f"Unsupported filter type for YDB: {filters.type}"
            raise ValueError(msg)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        import ydb

        if not embeddings:
            return 0, None

        if self.with_scalar_labels:
            if labels_data is None:
                msg = "labels_data is required when loading label-filter cases into YDB"
                raise ValueError(msg)
            if len(labels_data) != len(metadata):
                msg = "labels_data length must match metadata length"
                raise ValueError(msg)

        batch_size = 1000
        if self.with_scalar_labels:
            items_struct_type = ydb.StructType()
            items_struct_type.add_member("id", ydb.PrimitiveType.Uint64)
            items_struct_type.add_member("embedding", ydb.PrimitiveType.String)
            items_struct_type.add_member(YDB_LABEL_FIELD, ydb.PrimitiveType.Utf8)
            query = f"""
            DECLARE $items AS List<Struct<
                id: Uint64,
                embedding: String,
                {YDB_LABEL_FIELD}: Utf8
            >>;

            UPSERT INTO `{self.table_name}` (id, embedding, {YDB_LABEL_FIELD})
            SELECT id, embedding, {YDB_LABEL_FIELD}
            FROM AS_TABLE($items);
            """
        else:
            items_struct_type = ydb.StructType()
            items_struct_type.add_member("id", ydb.PrimitiveType.Uint64)
            items_struct_type.add_member("embedding", ydb.PrimitiveType.String)
            query = f"""
            DECLARE $items AS List<Struct<
                id: Uint64,
                embedding: String
            >>;

            UPSERT INTO `{self.table_name}` (id, embedding)
            SELECT id, embedding
            FROM AS_TABLE($items);
            """

        inserted = 0
        for offset in range(0, len(embeddings), batch_size):
            end = min(offset + batch_size, len(embeddings))
            if self.with_scalar_labels:
                items = [
                    {
                        "id": metadata[i],
                        "embedding": convert_vector_to_bytes(embeddings[i]),
                        YDB_LABEL_FIELD: labels_data[i],
                    }
                    for i in range(offset, end)
                ]
            else:
                items = [
                    {
                        "id": metadata[i],
                        "embedding": convert_vector_to_bytes(embeddings[i]),
                    }
                    for i in range(offset, end)
                ]
            self.pool.execute_with_retries(
                query,
                {"$items": (items, ydb.ListType(items_struct_type))},
            )
            inserted += len(items)

        return inserted, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        import ydb

        search_param = self.case_config.search_param()
        knn_function = search_param["knn_function"]
        sort_order = search_param["sort_order"]
        top_clusters = search_param["kmeans_tree_search_top_size"]

        use_index = self.case_config.create_index_after_load
        view_clause = f"VIEW {self.index_name}" if use_index else ""
        where_clause = f"\n        {self._where_clause}" if self._where_clause else ""
        label_declare = "\n        DECLARE $label AS Utf8;" if self._label_filter_value is not None else ""

        yql = f"""
        PRAGMA ydb.KMeansTreeSearchTopSize = "{top_clusters}";{label_declare}
        DECLARE $embedding AS String;

        SELECT id
        FROM `{self.table_name}` {view_clause}{where_clause}
        ORDER BY Knn::{knn_function}(embedding, $embedding) {sort_order}
        LIMIT {k};
        """

        params: dict[str, tuple] = {
            "$embedding": (
                convert_vector_to_bytes(query),
                ydb.PrimitiveType.String,
            ),
        }
        if self._label_filter_value is not None:
            params["$label"] = (self._label_filter_value, ydb.PrimitiveType.Utf8)

        result_sets = self.pool.execute_with_retries(yql, params)

        ids: list[int] = []
        for result_set in result_sets:
            for row in result_set.rows:
                ids.append(int(row["id"]))
        return ids
