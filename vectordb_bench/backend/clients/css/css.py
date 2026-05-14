import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from typing import Any, Final

from opensearchpy import OpenSearch
from packaging.version import Version
from packaging.version import parse as parse_version

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import CSSIndexConfig, CSS_Engine

log = logging.getLogger(__name__)

WAITING_FOR_REFRESH_SEC: Final[int] = 30
WAITING_FOR_FORCE_MERGE_SEC: Final[int] = 30
SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC: Final[int] = 30

VERSION_SPECIFIC_SETTING_RULES = [
    {
        "name": "knn.advanced.approximate_threshold",
        "applies": lambda version, _: version >= Version("3.0"),
        "value": lambda _: "-1",
    },
    {
        "name": "knn.derived_source.enabled",
        "applies": lambda version, _: version >= Version("3.0"),
        "value": lambda case_config: case_config.knn_derived_source_enabled,
    },
]

class CSSError(Exception):
    """Custom exception for CSS operations."""

class CSSSettingsManager:
    """Manages CSS cluster and index settings."""

    def __init__(self, client: OpenSearch, index_name:str) -> None:
        self.client = client
        self.index_name = index_name
    
    def apply_cluster_settings(self, settings: dict[str, Any], log_message: str = "Applied cluster settings", raise_on_error: bool = True) -> dict:
        """Apply cluster-level settings."""
        try:
            response = self.client.cluster.put_settings(body={"persistent": settings})
            log.info(log_message)
        except Exception as e:
            log.warning(f"Failed to apply cluster settings: {e}")
            if raise_on_error:
                error_msg = f"Cluster settings application failed: {e}"
                raise CSSError(error_msg) from e
            return {}
        else:
            return response

    def apply_index_settings(self, settings: dict[str, Any], log_message: str = "Applied index settings") -> dict:
        """Apply index-level settings."""
        try:
            response = self.client.indices.put_settings(index=self.index_name, body={"index": settings})
            log.info(log_message)
        except Exception as e:
            log.warning(f"Failed to apply index settings: {e}")
            error_msg = f"Index settings application failed: {e}"
            raise CSSError(error_msg) from e 
        else:
            return response

class BulkInsertManager:
    """Manages bulk insertion operations with chunking and parallelization."""

    def __init__(self, client: OpenSearch, index_name:str, case_config: CSSIndexConfig) -> None:
        self.client = client
        self.index_name = index_name
        self.case_config = case_config

    def prepare_bulk_data(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
        id_col_name: str,
        vector_col_name: str,
        label_col_name: str,
        with_scalar_labels: bool,
    ) -> list[dict[str, Any]]:
        """Prepare bulk actions for CSS OpenSearch bulk insert."""
        if len(embeddings) != len(metadata):
            error_msg = f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch"
            raise ValueError(error_msg)

        if with_scalar_labels and labels_data and len(labels_data) != len(embeddings):
            error_msg = f"Labels data ({len(labels_data)}) and embeddings ({len(embeddings)}) length mismatch"
            raise ValueError(error_msg)

        insert_data: list[dict[str, Any]] = []
        for i in range(len(embeddings)):
            index_data = {"index": {"_index": self.index_name}}
            if with_scalar_labels and self.case_config.use_routing and labels_data:
                index_data["routing"] = labels_data[i]        
            insert_data.append(index_data)
            other_data = {id_col_name: metadata[i], vector_col_name: embeddings[i]}
            if with_scalar_labels and labels_data:
                other_data[label_col_name] = labels_data[i]
            insert_data.append(other_data)
        return insert_data  
    
    def execute_single_client_insert(self, insert_data: list[dict[str, Any]]) -> tuple[int, Exception | None]:
        """Execute bulk insert with single client and retry logic."""
        max_retries = 10
        base_wait_seconds = 1
        wait_increment = 5

        for attempt in range(max_retries):
            try:
                response = self.client.bulk(body=insert_data)
                if response.get("errors"):
                    log.warning(f"Bulk insert had errors: {response}")
                return len(insert_data) // 2, None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait_seconds + wait_increment * attempt
                    log.warning(
                        f"Failed to insert data (attempt {attempt + 1}/{max_retries}): "
                        f"{self.index_name} error: {e!s}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    log.error(f"Failed to insert data after {max_retries} attempts: {self.index_name} error: {e!s}")
                    return 0, e

class SearchQueryBuilder:
    """Builds CSS KNN queries with proper configuration."""

    def __init__(self, case_config: CSSIndexConfig, vector_col_name: str) -> None:
        self.case_config = case_config
        self.vector_col_name = vector_col_name

        # Cache search parameters to avoid repeated computation
        self._cached_method_parameters = case_config.search_param()

    def build_knn_query(
        self, query_vector: list[float], k: int, filter_clause: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build a KNN query with optional filtering."""
        # Build KNN config using cached method_parameters
        knn_config: dict[str, Any] = {
            "vector": query_vector,
            "k": k,
            "method_parameters": self._cached_method_parameters
        }

        if filter_clause:
            knn_config["filter"] = filter_clause

        return {"size": k, "query": {"knn": {self.vector_col_name: knn_config}}}

    def build_search_kwargs(
        self, index_name: str, body: dict[str, Any], k: int, id_col_name: str, routing_key: str | None = None
    ) -> dict[str, Any]:
        """Build search kwargs with proper field selection."""
        # Directly build the kwargs dictionary without intermediate steps
        search_kwargs: dict[str, Any] = {
            "index": index_name,
            "body": body,
            "size": k,
            "_source": False,
            "stored_fields": ["_none_"],
            "docvalue_fields": [id_col_name],
            "preference": None,
        }
        if routing_key is not None:
            search_kwargs["routing"] = routing_key

        return search_kwargs

class CSS(VectorDB):    
    """CSS OpenSearch client implementation for VectorDBBench."""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict[str, Any],
        db_case_config: CSSIndexConfig,
        collection_name: str = "vdb_bench_index", # must be lowercase
        id_col_name: str = "id",
        label_col_name: str = "label",
        vector_col_name: str = "embedding",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the CSS OpenSearch client."""
        self.name = "CSS"
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_name = collection_name
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_col_name = vector_col_name
        self.with_scalar_labels = with_scalar_labels

        # Initialize client state
        self.client: OpenSearch | None = None
        self.filter: dict[str, Any] | None = None
        self.routing_key: str | None = None

        # Cache SearchQueryBuilder instance for better performance
        self._search_query_builder =SearchQueryBuilder(self.case_config, self.vector_col_name)

        # Pre-determine result extraction method to avoid per-query branching
        self._extract_result_ids = self._extract_ids_from_metadata

        log.info(f"CSS client config: {self.db_config}")
        log.info(f"CSS db case config: {self.case_config}")
        client = OpenSearch(**self.db_config)
        self._handle_index_initialization(client, drop_old)

    def _handle_index_initialization(self, client: OpenSearch, drop_old: bool) -> None:
        """Check, drop, create index, and perform post-creation setup."""
        if drop_old:
            log.info(f"CSS client drop old index: {self.index_name}")
            is_existed = client.indices.exists(index=self.index_name)
            if is_existed:
                client.indices.delete(index=self.index_name)
            self._create_index(client)
        else:
            is_existed = client.indices.exists(index=self.index_name)
            if not is_existed:
                self._create_index(client)
                log.info(f"CSS client create index: {self.index_name}")

    def need_normalize_cosine(self) -> bool:
        """Whether this database needs to normalize dataset to support COSINE metric."""
        return True

    def _get_cluster_version(self, client: OpenSearch) -> Version:
        """
        Return the CSS cluster version as a comparable Version object.
        Raises an exception if the version cannot be determined.
        """
        try:
            info = client.info()
            raw_version_str = info.get("version", {}).get("number", "")
            if not raw_version_str:
                raise ValueError("Received empty version string from OpenSearch")
            if raw_version_str.endswith("-SNAPSHOT"):
                raw_version_str = raw_version_str.replace("-SNAPSHOT", "")
            cluster_version = parse_version(raw_version_str)
            log.debug(f"Detected OpenSearch version: {cluster_version}")
            return cluster_version
        except Exception:
            log.exception("Failed to determine OpenSearch version")
            raise

    def _get_settings_manager(self, client: OpenSearch) -> CSSSettingsManager:
        """Get settings manager for the given client."""
        return CSSSettingsManager(client, self.index_name)

    def _get_version_specific_settings(self, cluster_version: Version) -> dict:
        """
        Builds and returns a dictionary of applicable version-specific settings.
        """
        version_specific_settings = {}
        for setting in VERSION_SPECIFIC_SETTING_RULES:
            if setting["applies"](cluster_version, self.case_config):
                name = setting["name"]
                value = setting["value"](self.case_config)
                version_specific_settings[name] = value
        return version_specific_settings

    def _build_vector_field_mapping(self) -> dict[str, Any]:
        """Build vector field mapping configuration for CSS OpenSearch.
        Uses index_param() as the single source of truth for method config.
        """
        log.info(f"Creating in-memory index with engine: {self.case_config.engine.value}")

        return {
            "type": "knn_vector",
            "dimension": self.dim,
            "method": self.case_config.index_param(),
        }

    def _get_bulk_manager(self, client: OpenSearch) -> BulkInsertManager:
        """Get bulk insertmanager for the given client."""
        return BulkInsertManager(client, self.index_name, self.case_config)

    def _create_index(self, client: OpenSearch) -> None:
        log.info(f"Creating index with number_of_replicas: {self.case_config.number_of_replicas}")
        log.info(f"Creating index with replication_type: {self.case_config.replication_type}")
        log.info(f"Creating index with knn_derived_source_enabled: {self.case_config.knn_derived_source_enabled}")
        log.info(f"Creating index with engine: {self.case_config.engine}")
        log.info(f"Creating index with metric type: {self.case_config.metric_type_name}")
        log.info(f"Creating index with memory_optimized_search: {self.case_config.memory_optimized_search}")
        log.info(f"All case_config parameters: {self.case_config.model_dump()}")

        settings_manager = self._get_settings_manager(client)
        cluster_settings = {
            "knn.algo_param.index_thread_qty": self.case_config.index_thread_qty,
            "knn.memory.circuit_breaker.limit": self.case_config.cb_threshold,
        }

        settings_manager.apply_cluster_settings(
            cluster_settings, "Successfully updated cluster settings for index creation",
            raise_on_error=False # Don't fail on unrecognized settings
        )

        cluster_version = self._get_cluster_version(client)

        # Base settings for CSS OpenSearch
        settings = {
            "index": {
                "knn": True,
                "number_of_shards": self.case_config.number_of_shards,
                "number_of_replicas": self.case_config.number_of_replicas,
            },
        }

        version_specific_settings = self._get_version_specific_settings(cluster_version)
        if version_specific_settings:
            log.info(f"Applying version-dependent settings: {version_specific_settings}")
            settings["index"].update(version_specific_settings)

        # Build properties mapping, excluding _id which is automatically handled by OpenSearch
        properties = {}

        # Only add id field to properties if it's not the special _id field
        if self.id_col_name != "_id":
            properties[self.id_col_name] = {"type": "integer"}
        properties[self.label_col_name] = {"type": "keyword"}
        properties[self.vector_col_name] = self._build_vector_field_mapping()

        mappings = {
            "_source": {"excludes": [self.vector_col_name], "recovery_source_excludes": [self.vector_col_name]},
            "properties": properties,
        }

        try:
            log.info(f"Creating index with settings: {settings}")
            log.info(f"Creating index with mappings: {mappings}")
            client.indices.create(
                index=self.index_name,
                body={"settings": settings, "mappings": mappings},
            )
        except Exception as e:
            log.warning(f"Failed to create index: {self.index_name} error: {e!s}")
            raise CSSError(f"Failed to create index: {self.index_name}: {e}") from e

    @contextmanager
    def init(self) -> None:
        """Connect to OpenSearch"""
        self.client = OpenSearch(**self.db_config)
        yield
        self.client = None

    def _prepare_bulk_data(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
    ) -> list[dict]:
        """Prepare the list of bulk actions for OpenSearch bulk insert."""
        bulk_manager = self._get_bulk_manager(self.client)
        return bulk_manager.prepare_bulk_data(
            list(embeddings),
            metadata,
            labels_data,
            self.id_col_name,
            self.vector_col_name,
            self.label_col_name,
            self.with_scalar_labels,
        )

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into the OpenSearch index."""
        assert self.client is not None, "should self.init() first"

        num_clients = self.case_config.number_of_indexing_clients or 1
        log.info(f"Number of indexing clients from case_config: {num_clients}")

        if num_clients <= 1:
            log.info("Using single client for data insertion")
            return self._insert_with_single_client(embeddings, metadata, labels_data)
        log.info(f"Using {num_clients} parallel clients for data insertion")
        return self._insert_with_multiple_clients(embeddings, metadata, num_clients, labels_data)

    def _insert_with_single_client(
        self,
        embeddings : Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
    ) -> tuple[int, Exception | None]:
        """Insert data using a single client with retry logic."""
        insert_data = self._prepare_bulk_data(embeddings, metadata, labels_data)
        bulk_manager = self._get_bulk_manager(self.client)
        return bulk_manager.execute_single_client_insert(insert_data)

    def _insert_with_multiple_clients(
        self,
        embeddings : Iterable[list[float]],
        metadata: list[int],
        num_clients: int,
        labels_data: list[str] | None = None,
    ) -> tuple[int, Exception | None]:
        """Insert data using multiple parallel clients for better performance."""
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor

        embeddings_list = list(embeddings)
        chunk_size = max(1, len(embeddings_list) // num_clients)
        chunks = []

        for i in range(0, len(embeddings_list), chunk_size):
            end = min(i + chunk_size, len(embeddings_list))
            chunk_labels = labels_data[i:end] if labels_data is not None else None
            chunks.append((embeddings_list[i:end], metadata[i:end], chunk_labels))
        clients = [OpenSearch(**self.db_config) for _ in range(min(num_clients, len(chunks)))]
        log.info(f"CSS using {len(clients)} parallel clients for data insertion")

        def insert_chunk(client_idx: int, chunk_idx: int):
            chunk_embeddings, chunk_metadata, chunk_labels_data = chunks[chunk_idx]
            client = clients[client_idx]
            insert_data = self._prepare_bulk_data(chunk_embeddings, chunk_metadata, chunk_labels_data)
            try:
                response = client.bulk(body=insert_data)
                log.info(f"Client {client_idx} added {len(response['items'])} documents")
                return len(chunk_embeddings), None
            except Exception as e:
                log.warning(f"Client {client_idx} failed to insert data: {e!s}")
                return 0, e

        results = []
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = [
                executor.submit(insert_chunk, chunk_idx % len(clients), chunk_idx) for chunk_idx in range(len(chunks))
            ]

            for future in concurrent.futures.as_completed(futures):
                count, error = future.result()
                results.append((count, error))

        for client in clients:
            with suppress(Exception):
                client.close()
        
        total_count = sum(count for count, _ in results)
        errors = [error for _, error in results if error is not None]

        if errors:
            log.warning("Some clients failed to insert data, retrying with single client")
            time.sleep(10)
            return self._insert_with_single_client(embeddings_list, metadata, labels_data)
        
        response = self.client.indices.stats(self.index_name)
        log.info(
            f"""Total document count in index after parallel insertion:
            {response['_all']['primaries']['indexing']['index_total']}."""
        )
        return (total_count, None)

    def _extract_ids_from_metadata(self, response: dict) -> list[int]:
        """Extract IDs from hit metadata when id_col_name is '_id'.
        This is faster than docvalue_fields because:
        1. Single dict access: h["_id"] vs h["fields"][id_col_name][0]
        2. No need to access nested "fields" dict
        3. No array indexing [0] overhead
        """
        try:
            result = []
            for hit in response["hits"]["hits"]:
                fields = hit.get('fields', {})
                id_values = fields.get('id', [])
                if id_values:
                    result.append(int(id_values[0]))
                else:
                    log.warning(f"Hit missing 'id' in fields: {hit.get('_id', 'unknown')}")
            return result
        except Exception as e:
            log.warning(f"Failed to extract IDs from metadata: {e}, response has {len(response.get('hits', {}).get('hits', []))} hits")
            return []

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.
        
        Args:
            query: Query vector.
            k: Number of similar embeddings to return.
            **kwargs: Additional arguments for search.
        
        Returns:
            List of IDs of the k most similar embeddings.
        Raises:
            Exception: If search operation fails.
        """
        assert self.client is not None, "should self.init() first"

        # Use cached SearchQueryBuilder instance
        body = self._search_query_builder.build_knn_query(query, k, self.filter)

        try:
            search_kwargs = self._search_query_builder.build_search_kwargs(
                self.index_name, body, k, self.id_col_name, self.routing_key
            )
            response = self.client.search(**search_kwargs)
            result = self._extract_result_ids(response)

            return result
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise CSSError(f"Failed to search: {self.index_name} error: {e}") from e

    def prepare_filter(self, filters: Filter) -> None:
        """Prepare filter conditions for search operations."""
        self.routing_key = None
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            self.filter = {"range": {self.id_col_name: {"gt": filters.int_value}}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {"term": {self.label_col_name: filters.label_value}}
            if self.case_config.use_routing:
                self.routing_key = filters.label_value
        else:
            msg = f"Filter type {filters.type} not supported for OpenSearch"
            log.error(f"Unsupported filter type: {filters.type}")
            raise ValueError(msg)

    def optimize(self, data_size: int | None = None) -> None:
        """Optimize the index for better search performance."""
        # Call refresh first to ensure that all segments are created
        self._refresh_index()
        if self.case_config.force_merge_enabled:
            self._do_force_merge()
            self._refresh_index()
        self._update_replicas()
        # Call refresh again to ensure that the index is ready after force merge.
        self._refresh_index()

    def _update_replicas(self):
        index_settings = self.client.indices.get_settings(index=self.index_name)
        current_number_of_replicas = int(index_settings[self.index_name]["settings"]["index"]["number_of_replicas"])
        log.info(
            f"Current Number of replicas are {current_number_of_replicas}"
            f" and changing the replicas to {self.case_config.number_of_replicas}"
        )
        settings_manager = self._get_settings_manager(self.client)
        log_message = f"Successfully updated number_of_replicas to {self.case_config.number_of_replicas}"
        settings_manager.apply_index_settings({"number_of_replicas": self.case_config.number_of_replicas}, log_message)
        self._wait_till_green()

    def _wait_till_green(self):
        log.info(f"Wait for index {self.index_name} to become green..")
        max_retries = 10 # Maximum retries before accepting yellow state
        retry_count = 0
        health = "unknown"
        while retry_count < max_retries:
            res = self.client.cat.indices(index=self.index_name, h="health", format="json")
            if not res:
                log.warning(f"Empty health response for {self.index_name}, retrying ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC)
                continue
            health = res[0]["health"]
            if health == "green":
                break
            log.info(f"The index {self.index_name} has health : {health} and is not green. Retrying ({retry_count + 1}/{max_retries})")
            time.sleep(SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC)
            retry_count += 1
        if health != "green":
            log.warning(f"Index {self.index_name} health is {health}, not green after {max_retries} retries.")
        else:
            log.info(f"Index {self.index_name} is green.")

    def _refresh_index(self):
        while True:
            try:
                log.info(f"Starting the Refresh Index for {self.index_name}")
                self.client.indices.refresh(index=self.index_name)
                break
            except Exception as e:
                log.info(
                    f"Refresh errored, retrying in {WAITING_FOR_REFRESH_SEC}s: {e}"
                )
                time.sleep(WAITING_FOR_REFRESH_SEC)
                continue

    def _do_force_merge(self):
        log.info(f"Updating the Index thread qty to {self.case_config.index_thread_qty_during_force_merge}.")
        settings_manager = self._get_settings_manager(self.client)
        cluster_settings = {"knn.algo_param.index_thread_qty": self.case_config.index_thread_qty_during_force_merge}
        log_message_cluster = (
            f"Successfully updated cluster index thread qty to {self.case_config.index_thread_qty_during_force_merge}"
        )
        settings_manager.apply_cluster_settings(cluster_settings, log_message_cluster)
        log.info("Updating the graph threshold to ensure that during merge we can do graph creation.")      
        log_message_index = "Successfully updated index approximate threshold to 0"
        output = settings_manager.apply_index_settings({"knn.advanced.approximate_threshold": "0"}, log_message_index)
        log.info(f"response of updating setting is: {output}")
        log.info(f"Starting force merge for index {self.index_name}")
        segments = self.case_config.number_of_segments
        force_merge_endpoint = f"/{self.index_name}/_forcemerge?max_num_segments={segments}&wait_for_completion=false"
        force_merge_task_id = self.client.transport.perform_request("POST", force_merge_endpoint)["task"]

        while True:
            time.sleep(WAITING_FOR_FORCE_MERGE_SEC)
            task_status = self.client.tasks.get(task_id=force_merge_task_id)
            if task_status.get("completed"):
                if task_status.get("error"):
                    raise CSSError(f"Force merge failed: {task_status['error']}")
                break
        log.info(f"Completed force merge for index {self.index_name}")
