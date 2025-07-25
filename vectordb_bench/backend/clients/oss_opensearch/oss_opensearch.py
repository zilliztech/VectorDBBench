import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from typing import Any, Final

from opensearchpy import OpenSearch

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import OSSOpenSearchIndexConfig, OSSOS_Engine

log = logging.getLogger(__name__)

WAITING_FOR_REFRESH_SEC: Final[int] = 30
WAITING_FOR_FORCE_MERGE_SEC: Final[int] = 30
SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC: Final[int] = 30


class OpenSearchError(Exception):
    """Custom exception for OpenSearch operations."""


class OpenSearchSettingsManager:
    """Manages OpenSearch cluster and index settings."""

    def __init__(self, client: OpenSearch, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def apply_cluster_settings(self, settings: dict[str, Any], log_message: str = "Applied cluster settings") -> dict:
        """Apply cluster-level settings."""
        try:
            response = self.client.cluster.put_settings(body={"persistent": settings})
            log.info(log_message)
        except Exception as e:
            log.warning(f"Failed to apply cluster settings: {e}")
            error_msg = f"Cluster settings application failed: {e}"
            raise OpenSearchError(error_msg) from e
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
            raise OpenSearchError(error_msg) from e
        else:
            return response


class BulkInsertManager:
    """Manages bulk insertion operations with chunking and parallelization."""

    def __init__(self, client: OpenSearch, index_name: str, case_config: OSSOpenSearchIndexConfig) -> None:
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
        """Prepare bulk actions for OpenSearch bulk insert."""
        if len(embeddings) != len(metadata):
            error_msg = f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch"
            raise ValueError(error_msg)

        if with_scalar_labels and labels_data and len(labels_data) != len(embeddings):
            error_msg = f"Labels data ({len(labels_data)}) and embeddings ({len(embeddings)}) length mismatch"
            raise ValueError(error_msg)

        insert_data: list[dict[str, Any]] = []
        for i in range(len(embeddings)):
            index_data = {"index": {"_index": self.index_name, id_col_name: metadata[i]}}
            if with_scalar_labels and self.case_config.use_routing and labels_data:
                index_data["routing"] = labels_data[i]
            insert_data.append(index_data)

            other_data = {vector_col_name: embeddings[i]}
            if with_scalar_labels and labels_data:
                other_data[label_col_name] = labels_data[i]
            insert_data.append(other_data)
        return insert_data

    def execute_single_client_insert(self, insert_data: list[dict[str, Any]]) -> tuple[int, Exception | None]:
        """Execute bulk insert with single client and retry logic."""
        try:
            response = self.client.bulk(body=insert_data)
            if response.get("errors"):
                log.warning(f"Bulk insert had errors: {response}")
            return len(insert_data) // 2, None
        except Exception as e:
            log.warning(f"Failed to insert data: {self.index_name} error: {e!s}")
            time.sleep(10)
            return self.execute_single_client_insert(insert_data)


class SearchQueryBuilder:
    """Builds OpenSearch KNN queries with proper configuration."""

    def __init__(self, case_config: OSSOpenSearchIndexConfig, vector_col_name: str) -> None:
        self.case_config = case_config
        self.vector_col_name = vector_col_name

    def build_knn_query(
        self, query_vector: list[float], k: int, filter_clause: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build a KNN query with optional filtering."""
        knn_config: dict[str, Any] = {
            "vector": query_vector,
            "k": k,
            "method_parameters": self.case_config.search_param(),
        }

        if filter_clause:
            knn_config["filter"] = filter_clause

        if self.case_config.use_quant:
            knn_config["rescore"] = {"oversample_factor": self.case_config.oversample_factor}

        return {"size": k, "query": {"knn": {self.vector_col_name: knn_config}}}

    def build_search_kwargs(
        self, index_name: str, body: dict[str, Any], k: int, id_col_name: str, routing_key: str | None = None
    ) -> dict[str, Any]:
        """Build search kwargs with proper field selection."""
        search_kwargs: dict[str, Any] = {
            "index": index_name,
            "body": body,
            "size": k,
            "_source": False,
            "preference": "_only_local" if self.case_config.number_of_shards == 1 else None,
            "routing": routing_key,
        }

        if id_col_name == "_id":
            search_kwargs["stored_fields"] = "_id"
        else:
            search_kwargs["docvalue_fields"] = [id_col_name]
            search_kwargs["stored_fields"] = "_none_"

        return search_kwargs


class OSSOpenSearch(VectorDB):
    """OpenSearch client implementation for VectorDBBench."""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict[str, Any],
        db_case_config: OSSOpenSearchIndexConfig,
        index_name: str = "vdb_bench_index",  # must be lowercase
        id_col_name: str = "_id",
        label_col_name: str = "label",
        vector_col_name: str = "embedding",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenSearch client."""
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_name = index_name
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_col_name = vector_col_name
        self.with_scalar_labels = with_scalar_labels

        # Initialize client state
        self.client: OpenSearch | None = None
        self.filter: dict[str, Any] | None = None
        self.routing_key: str | None = None

        log.info(f"OSS_OpenSearch client config: {self.db_config}")
        log.info(f"OSS_OpenSearch db case config: {self.case_config}")
        client = OpenSearch(**self.db_config)
        self._handle_index_initialization(client, drop_old)

    def _handle_index_initialization(self, client: OpenSearch, drop_old: bool) -> None:
        """Check, drop, create index, and perform post-creation setup."""
        if drop_old:
            log.info(f"OSS_OpenSearch client drop old index: {self.index_name}")
            is_existed = client.indices.exists(index=self.index_name)
            if is_existed:
                client.indices.delete(index=self.index_name)
            self._create_index(client)
        else:
            is_existed = client.indices.exists(index=self.index_name)
            if not is_existed:
                self._create_index(client)
                log.info(f"OSS_OpenSearch client create index: {self.index_name}")
            self._update_ef_search_before_search(client)
            self._load_graphs_to_memory(client)

    def need_normalize_cosine(self) -> bool:
        """Whether this database needs to normalize dataset to support COSINE metric."""
        return True

    def _get_settings_manager(self, client: OpenSearch) -> OpenSearchSettingsManager:
        """Get settings manager for the given client."""
        return OpenSearchSettingsManager(client, self.index_name)

    def _get_bulk_manager(self, client: OpenSearch) -> BulkInsertManager:
        """Get bulk insert manager for the given client."""
        return BulkInsertManager(client, self.index_name, self.case_config)

    def _create_index(self, client: OpenSearch) -> None:
        ef_search_value = self.case_config.efSearch
        log.info(f"Creating index with ef_search: {ef_search_value}")
        log.info(f"Creating index with number_of_replicas: {self.case_config.number_of_replicas}")
        log.info(f"Creating index with engine: {self.case_config.engine}")
        log.info(f"Creating index with metric type: {self.case_config.metric_type_name}")
        log.info(f"All case_config parameters: {self.case_config.__dict__}")

        settings_manager = self._get_settings_manager(client)
        cluster_settings = {
            "knn.algo_param.index_thread_qty": self.case_config.index_thread_qty,
            "knn.memory.circuit_breaker.limit": self.case_config.cb_threshold,
        }
        settings_manager.apply_cluster_settings(
            cluster_settings, "Successfully updated cluster settings for index creation"
        )
        settings = {
            "index": {
                "knn": True,
                "number_of_shards": self.case_config.number_of_shards,
                "number_of_replicas": self.case_config.number_of_replicas,
                "translog.flush_threshold_size": self.case_config.flush_threshold_size,
                "knn.advanced.approximate_threshold": "-1",
            },
            "refresh_interval": self.case_config.refresh_interval,
        }
        settings["index"]["knn.algo_param.ef_search"] = ef_search_value
        # Build properties mapping, excluding _id which is automatically handled by OpenSearch
        properties = {}

        # Only add id field to properties if it's not the special _id field
        if self.id_col_name != "_id":
            properties[self.id_col_name] = {"type": "integer", "store": True}

        properties[self.label_col_name] = {"type": "keyword"}
        properties[self.vector_col_name] = {
            "type": "knn_vector",
            "dimension": self.dim,
            "method": self.case_config.index_param(),
        }

        mappings = {
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
            raise e from None

    @contextmanager
    def init(self) -> None:
        """Connect to OpenSearch"""
        self.client = OpenSearch(**self.db_config)

        yield
        self.client = None
        del self.client

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
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
    ) -> tuple[int, Exception | None]:
        """Insert data using a single client with retry logic."""
        insert_data = self._prepare_bulk_data(embeddings, metadata, labels_data)
        bulk_manager = self._get_bulk_manager(self.client)
        return bulk_manager.execute_single_client_insert(insert_data)

    def _insert_with_multiple_clients(
        self,
        embeddings: Iterable[list[float]],
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
            chunks.append((embeddings_list[i:end], metadata[i:end], labels_data[i:end]))
        clients = [OpenSearch(**self.db_config) for _ in range(min(num_clients, len(chunks)))]
        log.info(f"OSS_OpenSearch using {len(clients)} parallel clients for data insertion")

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
            return self._insert_with_single_client(embeddings, metadata, labels_data)

        response = self.client.indices.stats(self.index_name)
        log.info(
            f"""Total document count in index after parallel insertion:
                {response['_all']['primaries']['indexing']['index_total']}""",
        )

        return (total_count, None)

    def _update_ef_search_before_search(self, client: OpenSearch):
        ef_search_value = self.case_config.efSearch

        try:
            index_settings = client.indices.get_settings(index=self.index_name)
            current_ef_search = (
                index_settings.get(self.index_name, {})
                .get("settings", {})
                .get("index", {})
                .get("knn.algo_param", {})
                .get("ef_search")
            )

            if current_ef_search != str(ef_search_value):
                settings_manager = self._get_settings_manager(client)
                log_message = f"Successfully updated ef_search to {ef_search_value} before search"
                settings_manager.apply_index_settings({"knn.algo_param.ef_search": ef_search_value}, log_message)
            log.info(f"Current engine: {self.case_config.engine}")
            log.info(f"Current metric_type: {self.case_config.metric_type_name}")

        except Exception as e:
            log.warning(f"Failed to update ef_search parameter before search: {e}")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: Filter | None = None,
        **kwargs,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(Filter, optional): filtering expression to filter the data while searching.

        Returns:
            list[int]: list of k most similar ids to the query embedding.
        """
        assert self.client is not None, "should self.init() first"

        search_query_builder = SearchQueryBuilder(self.case_config, self.vector_col_name)
        body = search_query_builder.build_knn_query(query, k, self.filter)

        try:
            search_kwargs = search_query_builder.build_search_kwargs(
                self.index_name, body, k, self.id_col_name, self.routing_key
            )
            response = self.client.search(**search_kwargs)

            log.debug(f"Search took: {response['took']}")
            log.debug(f"Search shards: {response['_shards']}")
            log.debug(f"Search hits total: {response['hits']['total']}")
            try:
                if self.id_col_name == "_id":
                    # Get _id directly from hit metadata
                    result_ids = []
                    for h in response["hits"]["hits"]:
                        if (doc_id := h.get("_id")) is not None:
                            result_ids.append(int(doc_id))
                        else:
                            log.warning(f"Hit missing _id in final extraction: {h}")
                else:
                    # Get custom id field from docvalue fields
                    result_ids = [int(h["fields"][self.id_col_name][0]) for h in response["hits"]["hits"]]

            except Exception:
                # empty results
                return []
            else:
                return result_ids
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise e from None

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
        self._update_ef_search()
        # Call refresh first to ensure that all segments are created
        self._refresh_index()
        if self.case_config.force_merge_enabled:
            self._do_force_merge()
            self._refresh_index()
        self._update_replicas()
        # Call refresh again to ensure that the index is ready after force merge.
        self._refresh_index()
        # ensure that all graphs are loaded in memory and ready for search
        self._load_graphs_to_memory(self.client)

    def _update_ef_search(self):
        ef_search_value = self.case_config.efSearch
        settings_manager = self._get_settings_manager(self.client)
        log_message = f"Successfully updated ef_search to {ef_search_value}"
        settings_manager.apply_index_settings({"knn.algo_param.ef_search": ef_search_value}, log_message)
        log.info(f"Current engine: {self.case_config.engine}")
        log.info(f"Current metric_type: {self.case_config.metric_type}")

    def _update_replicas(self):
        index_settings = self.client.indices.get_settings(index=self.index_name)
        current_number_of_replicas = int(index_settings[self.index_name]["settings"]["index"]["number_of_replicas"])
        log.info(
            f"Current Number of replicas are {current_number_of_replicas}"
            f" and changing the replicas to {self.case_config.number_of_replicas}"
        )
        settings_manager = self._get_settings_manager(self.client)
        log_message = f"Successfully updated number of replicas to {self.case_config.number_of_replicas}"
        settings_manager.apply_index_settings({"number_of_replicas": self.case_config.number_of_replicas}, log_message)
        self._wait_till_green()

    def _wait_till_green(self):
        log.info("Wait for index to become green..")
        while True:
            res = self.client.cat.indices(index=self.index_name, h="health", format="json")
            health = res[0]["health"]
            if health == "green":
                break
            log.info(f"The index {self.index_name} has health : {health} and is not green. Retrying")
            time.sleep(SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC)
        log.info(f"Index {self.index_name} is green..")

    def _refresh_index(self):
        log.debug(f"Starting refresh for index {self.index_name}")
        while True:
            try:
                log.info("Starting the Refresh Index..")
                self.client.indices.refresh(index=self.index_name)
                break
            except Exception as e:
                log.info(
                    f"Refresh errored out. Sleeping for {WAITING_FOR_REFRESH_SEC} sec and then Retrying : {e}",
                )
                time.sleep(WAITING_FOR_REFRESH_SEC)
                continue
        log.debug(f"Completed refresh for index {self.index_name}")

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
            if task_status["completed"]:
                break
        log.info(f"Completed force merge for index {self.index_name}")

    def _load_graphs_to_memory(self, client: OpenSearch):
        if self.case_config.engine != OSSOS_Engine.lucene:
            log.info("Calling warmup API to load graphs into memory")
            warmup_endpoint = f"/_plugins/_knn/warmup/{self.index_name}"
            client.transport.perform_request("GET", warmup_endpoint)
