import logging
import os
from contextlib import contextmanager
from typing import Any

import pandas as pd
from doris_vector_search import AuthOptions, DorisVectorClient, IndexOptions, LoadOptions

from ..api import MetricType, VectorDB
from .config import DorisCaseConfig

log = logging.getLogger(__name__)


class Doris(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DorisCaseConfig,
        collection_name: str | None = None,
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Doris"
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim
        self.search_fn = db_case_config.search_param()["metric_fn"]
        # Prefer provided collection_name; otherwise fallback to a simple default
        # e.g. l2_distance128, inner_product128
        self.table_name = collection_name if collection_name else (self.search_fn + str(dim))

        # Store connection configuration for lazy initialization
        self.auth_options = AuthOptions(
            host=db_config.get("host", "127.0.0.1"),
            query_port=db_config.get("port", 9030),
            http_port=db_config.get("http_port", 8030),
            user=db_config.get("user", "root"),
            password=db_config.get("password", ""),
        )

        # Configure load options
        self.load_options = None
        if hasattr(db_case_config, "stream_load_rows_per_batch") and db_case_config.stream_load_rows_per_batch:
            self.load_options = LoadOptions(batch_size=db_case_config.stream_load_rows_per_batch)

        # Store database name for lazy initialization
        self.database_name = db_config.get("database", "test")

        # Initialize client and table as None (lazy initialization)
        self.client = None
        self.table = None
        self._client_pid: int | None = None

        if drop_old:
            self._drop_table()
            self._create_table()

    def _ensure_client_initialized(self):
        """Ensure the client is initialized and bound to the current process.

        Multiprocessing will pickle the DB wrapper. Any existing mysql connection or
        table cursor cached from a different PID must be discarded and recreated here.
        """
        cur_pid = os.getpid()

        need_new_client = False
        if self.client is None:
            need_new_client = True
        else:
            # If client was created in another process or connection is not usable, recreate it
            try:
                # different process
                if self._client_pid is None or self._client_pid != cur_pid:
                    need_new_client = True
                else:
                    # check connection health if available
                    conn = getattr(self.client, "connection", None)
                    if conn is None or not getattr(conn, "is_connected", lambda: False)():
                        need_new_client = True
            except Exception:
                need_new_client = True

        if need_new_client:
            # Drop any table cached from another PID (its cursors are not valid across processes)
            self.table = None

            # Recreate client and set sessions
            self.client = DorisVectorClient(
                database=self.database_name,
                auth_options=self.auth_options,
                load_options=self.load_options,
            )

            if hasattr(self.case_config, "session_vars") and self.case_config.session_vars:
                self.client.with_sessions(self.case_config.session_vars)

            self._client_pid = cur_pid

            # Re-open table in this process to ensure fresh cursors
            try:
                self.table = self.client.open_table(self.table_name)
                if hasattr(self.table, "index_options") and self.table.index_options:
                    self.table.index_options.dim = self.dim
                    if self.search_fn.startswith("inner_product"):
                        self.table.index_options.metric_type = "inner_product"
                    else:
                        self.table.index_options.metric_type = "l2_distance"
            except Exception:
                # Table might not exist yet; leave it to ready_to_load
                self.table = None

    @contextmanager
    def init(self):
        try:
            self._ensure_client_initialized()
            # Open or create the table
            if not self.table:
                try:
                    # Try to open existing table
                    self.table = self.client.open_table(self.table_name)
                    # Avoid SHOW CREATE TABLE parsing in SDK by setting dim/metric directly
                    try:
                        if hasattr(self.table, "index_options") and self.table.index_options:
                            self.table.index_options.dim = self.dim
                            # Set metric_type according to current case
                            if self.search_fn.startswith("inner_product"):
                                self.table.index_options.metric_type = "inner_product"
                            else:
                                self.table.index_options.metric_type = "l2_distance"
                    except Exception:
                        log.exception("Failed to update index options for table: %s", self.table_name)
                except Exception:
                    # Table doesn't exist, will be created in ready_to_load
                    self.table = None
            yield
        finally:
            # Clean up if needed
            pass

    def _drop_table(self):
        try:
            self._ensure_client_initialized()
            self.client.drop_table(self.table_name)
        except Exception:
            log.exception("Failed to drop table: %s", self.table_name)
            raise

    def _create_table(self):
        """Create the table using doris-vector-search library"""
        try:
            self._ensure_client_initialized()
            sample_data = pd.DataFrame([{"id": 1, "embedding": [0.0] * self.dim}])

            index_options = None
            if not getattr(self.case_config, "no_index", False):
                index_options = self._build_index_options()

            self._create_table_with_options(sample_data, index_options)
            log.info("Successfully created table %s", self.table_name)

        except Exception:
            log.exception("Failed to create table: %s", self.table_name)
            raise

    def _build_index_options(self) -> IndexOptions | None:
        index_param = self.case_config.index_param()
        index_options = IndexOptions()

        applied, not_applied = {}, {}
        for key, value in index_param.items():
            attr_name = key
            if hasattr(index_options, attr_name):
                try:
                    setattr(index_options, attr_name, value)
                    applied[key] = value
                except Exception:
                    not_applied[key] = value
            else:
                not_applied[key] = value

        log.info(
            "Index options prepared: applied_props=%s not_applied_props=%s",
            applied,
            not_applied,
        )

        log.info("Creating table %s with index %s", self.table_name, index_param)
        return index_options

    def _create_table_with_options(self, sample_data: pd.DataFrame, index_options: IndexOptions | None) -> None:
        create_index = not getattr(self.case_config, "no_index", False)
        if not create_index:
            log.info("Creating table %s without ANN index", self.table_name)

        self.table = self.client.create_table(
            self.table_name,
            sample_data,
            create_index=create_index,
            index_options=index_options,
            overwrite=True,
            insert_data=False,
        )

        try:
            if hasattr(self.table, "index_options") and self.table.index_options:
                self.table.index_options.dim = self.dim
                if self.search_fn.startswith("inner_product"):
                    self.table.index_options.metric_type = "inner_product"
                else:
                    self.table.index_options.metric_type = "l2_distance"
                if (
                    index_options
                    and hasattr(index_options, "properties")
                    and isinstance(index_options.properties, dict)
                ):
                    for key, value in index_options.properties.items():
                        if hasattr(self.table.index_options, key):
                            try:
                                setattr(self.table.index_options, key, value)
                            except Exception:
                                log.debug("Skip setting index_options.%s at runtime", key)
        except Exception:
            log.exception("Failed to adjust index options for table: %s", self.table_name)

    def ready_to_load(self) -> bool:
        self._ensure_client_initialized()
        if not self.table:
            self._create_table()
        return True

    def optimize(self, data_size: int | None = None) -> None:
        log.info("Optimization completed using doris-vector-search library")

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        if self.case_config.metric_type == MetricType.COSINE:
            log.info("cosine dataset need normalize.")
            return True

        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings using doris-vector-search library."""
        try:
            self._ensure_client_initialized()
            # Prepare data in pandas DataFrame format
            data = pd.DataFrame([{"id": metadata[i], "embedding": embeddings[i]} for i in range(len(embeddings))])

            msg = f"Inserting {len(embeddings)} embeddings into table {self.table_name}"
            log.info(msg)

            # Add data to the table
            self.table.add(data)

            return len(metadata), None

        except Exception as e:
            msg = "Failed to insert embeddings"
            log.exception(msg)
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        try:
            self._ensure_client_initialized()
            # Map metric functions to doris-vector-search metric types
            metric_type = "l2_distance"
            if self.search_fn.startswith("inner_product"):
                metric_type = "inner_product"
            elif self.search_fn.startswith("l2_distance"):
                metric_type = "l2_distance"

            # Perform search using doris-vector-search
            search_query = self.table.search(query, metric_type=metric_type).limit(k).select(["id"])

            # Apply filters if provided
            if filters and "id" in filters:
                if self.search_fn.startswith("inner_product"):
                    where_clause = f"id >= {filters['id']}"
                    search_query = search_query.where(where_clause)
                else:
                    where_clause = f"id < {filters['id']}"
                    search_query = search_query.where(where_clause)

            # Execute and get results
            results_df = search_query.to_pandas()
            return results_df["id"].tolist()

        except Exception:
            msg = "Search embedding failed"
            log.exception(msg)
            return []

    def search_embedding_range(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        distance: float | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        try:
            self._ensure_client_initialized()
            # Map metric functions to doris-vector-search metric types
            metric_type = "l2_distance"
            if self.search_fn.startswith("inner_product"):
                metric_type = "inner_product"
            elif self.search_fn.startswith("l2_distance"):
                metric_type = "l2_distance"

            # Perform range search using doris-vector-search
            search_query = self.table.search(query, metric_type=metric_type).select(["id"])

            # Apply distance range
            if distance is not None:
                if self.search_fn.startswith("inner_product"):
                    adjusted_distance = distance - 0.000001
                    search_query = search_query.distance_range(lower_bound=adjusted_distance)
                else:
                    adjusted_distance = distance + 0.000001
                    search_query = search_query.distance_range(upper_bound=adjusted_distance)

            # Execute and get results
            results_df = search_query.to_pandas()
            return results_df["id"].tolist()

        except Exception:
            msg = "Range search failed"
            log.exception(msg)
            return []

    def search_embedding_compound(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        distance: float | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        try:
            self._ensure_client_initialized()
            # Map metric functions to doris-vector-search metric types
            metric_type = "l2_distance"
            if self.search_fn.startswith("inner_product"):
                metric_type = "inner_product"
            elif self.search_fn.startswith("l2_distance"):
                metric_type = "l2_distance"

            # Perform compound search using doris-vector-search
            search_query = self.table.search(query, metric_type=metric_type).limit(k).select(["id"])

            # Apply distance range
            if distance is not None:
                if self.search_fn.startswith("inner_product"):
                    adjusted_distance = distance - 0.000001
                    search_query = search_query.distance_range(lower_bound=adjusted_distance)
                else:
                    adjusted_distance = distance + 0.000001
                    search_query = search_query.distance_range(upper_bound=adjusted_distance)

            # Execute and get results
            results_df = search_query.to_pandas()
            return results_df["id"].tolist()

        except Exception:
            msg = "Compound search failed"
            log.exception(msg)
            return []

    def search_distance(self, query: list[float], target_id: int | None = None):
        try:
            self._ensure_client_initialized()
            metric_type = self.search_fn
            if metric_type.endswith("_approximate"):
                metric_type = metric_type.replace("_approximate", "")

            search_metric = "inner_product" if metric_type.startswith("inner_product") else "l2_distance"
            where_clause = f"id = {target_id}"
            search_query = self.table.search(query, metric_type=search_metric).where(where_clause).select(["id"])
            results_df = search_query.to_pandas()
        except Exception:
            msg = "Distance search failed"
            log.exception(msg)
            return []

        # For now, return a placeholder distance
        # The exact distance calculation would need custom SQL or library support
        return [0.0] if not results_df.empty else []

    def search_embedding_exact(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        try:
            self._ensure_client_initialized()
            # Use exact search by removing approximate suffix
            metric_type = self.search_fn
            if metric_type.endswith("_approximate"):
                metric_type = metric_type.replace("_approximate", "")

            # Map to doris-vector-search metric types
            search_metric = "inner_product" if metric_type.startswith("inner_product") else "l2_distance"

            # Perform exact search
            search_query = self.table.search(query, metric_type=search_metric).limit(k).select(["id"])

            # Apply filters if provided
            if filters and "id" in filters:
                if metric_type.startswith("inner_product"):
                    where_clause = f"id >= {filters['id']}"
                    search_query = search_query.where(where_clause)
                else:
                    where_clause = f"id < {filters['id']}"
                    search_query = search_query.where(where_clause)

            # Execute and get results
            results_df = search_query.to_pandas()
            return results_df["id"].tolist()

        except Exception:
            msg = "Exact search failed"
            log.exception(msg)
            return []
