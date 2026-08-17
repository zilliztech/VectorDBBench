import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from opensearchpy import OpenSearch

from vectordb_bench import config
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AWSOpenSearchIndexConfig, AWSOS_Engine

log = logging.getLogger(__name__)

WAITING_FOR_REFRESH_SEC = 30
WAITING_FOR_FORCE_MERGE_SEC = 30
SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC = 30
BULK_MAX_ATTEMPTS = 30
BULK_INITIAL_RETRY_DELAY_SEC = 2
BULK_MAX_RETRY_DELAY_SEC = 60


class OpenSearchBulkInsertError(RuntimeError):
    non_retryable = True


class AWSOpenSearch(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AWSOpenSearchIndexConfig,
        index_name: str = "vdb_bench_index",  # must be lowercase
        id_col_name: str = "_id",
        label_col_name: str = "label",
        vector_col_name: str = "embedding",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_name = index_name
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_col_name = vector_col_name
        self.with_scalar_labels = with_scalar_labels

        log.info(f"AWS_OpenSearch client config: {self.db_config}")
        log.info(f"AWS_OpenSearch db case config : {self.case_config}")
        self._is_serverless = ".aoss." in self.db_config.get("hosts", [{}])[0].get("host", "")
        client = OpenSearch(**self.db_config)
        if drop_old:
            log.info(f"AWS_OpenSearch client drop old index: {self.index_name}")
            is_existed = client.indices.exists(index=self.index_name)
            if is_existed:
                client.indices.delete(index=self.index_name)
            self._create_index(client)
        else:
            is_existed = client.indices.exists(index=self.index_name)
            if not is_existed:
                self._create_index(client)
                log.info(f"AWS_OpenSearch client create index: {self.index_name}")

            self._update_ef_search_before_search(client)
            self._load_graphs_to_memory(client)

    def _create_index(self, client: OpenSearch) -> None:
        self._log_index_creation_info()
        self._configure_cluster_settings(client)
        settings = self._build_index_settings()
        vector_field_config = self._build_vector_field_config()
        mappings = self._build_mappings(vector_field_config)
        self._create_opensearch_index(client, settings, mappings)

    def _log_index_creation_info(self) -> None:
        log.info(f"Creating index with ef_search: {self.case_config.ef_search}")
        log.info(f"Creating index with number_of_replicas: {self.case_config.number_of_replicas}")
        log.info(f"Creating index with engine: {self.case_config.engine}")
        log.info(f"Creating index with metric type: {self.case_config.metric_type_name}")
        log.info(f"All case_config parameters: {self.case_config.__dict__}")

    def _configure_cluster_settings(self, client: OpenSearch) -> None:
        if self._is_serverless:
            log.info("Skipping cluster settings for OpenSearch Serverless")
            return
        cluster_settings_body = {
            "persistent": {
                "knn.algo_param.index_thread_qty": self.case_config.index_thread_qty,
                "knn.memory.circuit_breaker.limit": self.case_config.cb_threshold,
            }
        }
        client.cluster.put_settings(body=cluster_settings_body)

    def _build_index_settings(self) -> dict:
        settings = {
            "index": {
                "knn": True,
                "number_of_shards": self.case_config.number_of_shards,
                "number_of_replicas": self.case_config.number_of_replicas,
            },
        }
        if not self._is_serverless:
            settings["index"]["translog.flush_threshold_size"] = self.case_config.flush_threshold_size
            settings["index"]["knn.advanced.approximate_threshold"] = "-1"
            settings["index"]["knn.algo_param.ef_search"] = self.case_config.ef_search
            settings["refresh_interval"] = self.case_config.refresh_interval
        return settings

    def _build_vector_field_config(self) -> dict:
        method_config = self.case_config.index_param()
        log.info(f"Raw method config from index_param(): {method_config}")

        if self.case_config.engine == AWSOS_Engine.s3vector:
            method_config = {"engine": "s3vector"}

        # OpenSearch Serverless does not support 'engine' or 'encoder' in method config
        if self._is_serverless and "engine" in method_config:
            space_type = method_config.pop("space_type", self.case_config.parse_metric())
            method_config.pop("engine", None)
            if "parameters" in method_config:
                method_config["parameters"].pop("encoder", None)
            vector_field_config = {
                "type": "knn_vector",
                "dimension": self.dim,
                "space_type": space_type,
                "method": method_config,
            }
            log.info(f"Serverless vector field config: {vector_field_config}")
            return vector_field_config

        if self.case_config.on_disk:
            space_type = self.case_config.parse_metric()
            vector_field_config = {
                "type": "knn_vector",
                "dimension": self.dim,
                "space_type": space_type,
                "data_type": "float",
                "mode": "on_disk",
                "compression_level": "32x",
                "method": method_config,
            }
            log.info("Using on-disk vector configuration with compression_level: 32x")
        else:
            vector_field_config = {
                "type": "knn_vector",
                "dimension": self.dim,
                "method": method_config,
            }

        if self.case_config.on_disk:
            log.info(f"Final on-disk vector field config: {vector_field_config}")
        elif self.case_config.engine == AWSOS_Engine.s3vector:
            space_type = self.case_config.parse_metric()
            vector_field_config["space_type"] = space_type
            vector_field_config["method"] = {"engine": "s3vector"}
            log.info(f"Final vector field config for s3vector: {vector_field_config}")
        else:
            log.info(f"Standard vector field config: {vector_field_config}")

        return vector_field_config

    def _build_mappings(self, vector_field_config: dict) -> dict:
        if self.case_config.engine == AWSOS_Engine.s3vector:
            mappings = {
                "properties": {
                    self.label_col_name: {"type": "keyword"},
                    self.vector_col_name: vector_field_config,
                },
            }
            log.info("Using simplified mappings for s3vector engine (no _source configuration)")
        else:
            mappings = {
                "_source": {"excludes": [self.vector_col_name], "recovery_source_excludes": [self.vector_col_name]},
                "properties": {
                    self.label_col_name: {"type": "keyword"},
                    self.vector_col_name: vector_field_config,
                },
            }
            log.info("Using standard mappings with _source configuration for non-s3vector engines")

        # Serverless stores the benchmark id in a dedicated numeric field (custom _id
        # is not supported). Map it as a long so NumGE range filters work.
        if self._is_serverless:
            mappings["properties"]["id"] = {"type": "long"}

        return mappings

    def _create_opensearch_index(self, client: OpenSearch, settings: dict, mappings: dict) -> None:
        try:
            log.info(f"Creating index with settings: {settings}")
            log.info(f"Creating index with mappings: {mappings}")

            if self.case_config.engine == AWSOS_Engine.s3vector:
                method_in_mappings = mappings["properties"][self.vector_col_name]["method"]
                log.info(f"Final method config being sent to OpenSearch: {method_in_mappings}")

            client.indices.create(
                index=self.index_name,
                body={"settings": settings, "mappings": mappings},
            )

            if self.case_config.engine == AWSOS_Engine.s3vector:
                self._verify_s3vector_index_config(client)

        except Exception as e:
            log.warning(f"Failed to create index: {self.index_name} error: {e!s}")
            raise e from None

    def _verify_s3vector_index_config(self, client: OpenSearch) -> None:
        try:
            actual_mapping = client.indices.get_mapping(index=self.index_name)
            actual_method = actual_mapping[self.index_name]["mappings"]["properties"][self.vector_col_name]["method"]
            log.info(f"Actual method config in created index: {actual_method}")
        except Exception as e:
            log.warning(f"Failed to verify index configuration: {e}")

    @contextmanager
    def init(self) -> None:
        """connect to opensearch"""
        self.client = OpenSearch(**self.db_config)

        yield
        self.client = None
        del self.client

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert the embeddings to the opensearch."""
        assert self.client is not None, "should self.init() first"

        num_clients = self.case_config.number_of_indexing_clients or 1
        log.info(f"Number of indexing clients from case_config: {num_clients}")

        # OpenSearch Serverless requires the single-client path because it does not
        # support custom _id and needs the benchmark id stored in _source.
        if self._is_serverless:
            log.info("Using single client for data insertion (OpenSearch Serverless)")
            return self._insert_with_single_client(embeddings, metadata, labels_data)

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
    ) -> tuple[int, Exception]:
        embeddings_list = list(embeddings)
        batch_size = config.NUM_PER_BATCH if self._is_serverless else len(embeddings_list)
        if self._is_serverless and batch_size <= 0:
            raise ValueError("NUM_PER_BATCH must be greater than 0 for OpenSearch Serverless")
        total_inserted = 0

        for i in range(0, len(embeddings_list), batch_size):
            batch_embeddings = embeddings_list[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size]
            batch_labels = labels_data[i : i + batch_size] if labels_data else None

            insert_data = []
            for j in range(len(batch_embeddings)):
                if self._is_serverless:
                    index_data = {"index": {"_index": self.index_name}}
                else:
                    index_data = {"index": {"_index": self.index_name, self.id_col_name: batch_metadata[j]}}

                if self.with_scalar_labels and self.case_config.use_routing and batch_labels is not None:
                    index_data["routing"] = batch_labels[j]
                insert_data.append(index_data)

                other_data = {self.vector_col_name: batch_embeddings[j]}
                if self._is_serverless:
                    other_data["id"] = batch_metadata[j]
                if self.with_scalar_labels and batch_labels is not None:
                    other_data[self.label_col_name] = batch_labels[j]
                insert_data.append(other_data)

            inserted, error = self._execute_bulk_with_retries(
                self.client,
                insert_data,
                f"index {self.index_name}",
            )
            total_inserted += inserted
            if error is not None:
                raise error

        return total_inserted, None

    @staticmethod
    def _parse_bulk_response(
        response: dict[str, Any],
        insert_data: list[dict[str, Any]],
    ) -> tuple[int, list[dict[str, Any]], list[str]]:
        expected_count = len(insert_data) // 2
        if not response.get("errors"):
            return expected_count, [], []

        items = response.get("items")
        if not isinstance(items, list):
            return 0, insert_data, ["response did not contain an items list"]

        success_count = 0
        failed_data = []
        failure_samples = []
        for position in range(expected_count):
            item = items[position] if position < len(items) else None
            if isinstance(item, dict) and len(item) == 1:
                operation, result = next(iter(item.items()))
                if isinstance(result, dict):
                    status = result.get("status")
                    if isinstance(status, int) and 200 <= status < 300 and "error" not in result:
                        success_count += 1
                        continue
                    error = result.get("error", "unknown")
                    failure_samples.append(
                        f"item[{position}] {operation} id={result.get('_id', 'unknown')} "
                        f"status={status or 'unknown'} error={error}"
                    )
                else:
                    failure_samples.append(f"item[{position}] {operation}=malformed")
            else:
                failure_samples.append(f"item[{position}]=malformed")
            failed_data.extend(insert_data[position * 2 : position * 2 + 2])

        return success_count, failed_data, failure_samples

    def _execute_bulk_with_retries(
        self,
        client: OpenSearch,
        insert_data: list[dict[str, Any]],
        context: str,
    ) -> tuple[int, Exception | None]:
        pending_data = insert_data
        total_inserted = 0

        for attempt in range(1, BULK_MAX_ATTEMPTS + 1):
            response: Any = None
            request_error = None
            try:
                response = client.bulk(body=pending_data)
            except Exception as e:
                request_error = e

            if request_error is None and not isinstance(response, dict):
                request_error = TypeError("OpenSearch bulk response was not an object")

            if request_error is not None:
                if attempt < BULK_MAX_ATTEMPTS:
                    retry_delay = min(
                        BULK_INITIAL_RETRY_DELAY_SEC * (2 ** (attempt - 1)),
                        BULK_MAX_RETRY_DELAY_SEC,
                    )
                    log.warning(
                        f"Bulk request failed for {context}; next attempt {attempt + 1}/{BULK_MAX_ATTEMPTS} "
                        f"in {retry_delay}s: {request_error!s}"
                    )
                    time.sleep(retry_delay)
                    continue
                error = OpenSearchBulkInsertError(
                    f"Bulk request failed for {context} after {BULK_MAX_ATTEMPTS} attempts; "
                    f"successful={total_inserted}: {request_error!s}"
                )
                log.error(str(error))
                return total_inserted, error

            inserted, failed_data, failure_samples = self._parse_bulk_response(response, pending_data)
            total_inserted += inserted
            if not failed_data:
                return total_inserted, None

            failed_count = len(failed_data) // 2
            sample_summary = "; ".join(failure_samples[:3])
            if attempt < BULK_MAX_ATTEMPTS:
                retry_delay = min(
                    BULK_INITIAL_RETRY_DELAY_SEC * (2 ** (attempt - 1)),
                    BULK_MAX_RETRY_DELAY_SEC,
                )
                log.warning(
                    f"Bulk response for {context} rejected {failed_count} documents; "
                    f"next attempt {attempt + 1}/{BULK_MAX_ATTEMPTS} in {retry_delay}s; "
                    f"{sample_summary}"
                )
                pending_data = failed_data
                time.sleep(retry_delay)
                continue

            error = OpenSearchBulkInsertError(
                f"Bulk insert for {context} left {failed_count} documents uninserted after "
                f"{BULK_MAX_ATTEMPTS} attempts; successful={total_inserted}; {sample_summary}"
            )
            log.error(str(error))
            return total_inserted, error

        error = OpenSearchBulkInsertError(f"Bulk insert for {context} exhausted its retry loop")
        return total_inserted, error

    def _insert_with_multiple_clients(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        num_clients: int,
        labels_data: list[str] | None = None,
    ) -> tuple[int, Exception]:
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor

        embeddings_list = list(embeddings)
        chunk_size = max(1, len(embeddings_list) // num_clients)
        chunks = []

        for i in range(0, len(embeddings_list), chunk_size):
            end = min(i + chunk_size, len(embeddings_list))
            chunk_labels = labels_data[i:end] if labels_data is not None else None
            chunks.append((embeddings_list[i:end], metadata[i:end], chunk_labels))

        clients = []
        for _ in range(min(num_clients, len(chunks))):
            client = OpenSearch(**self.db_config)
            clients.append(client)

        log.info(f"AWS_OpenSearch using {len(clients)} parallel clients for data insertion")

        def insert_chunk(client_idx: int, chunk_idx: int):
            chunk_embeddings, chunk_metadata, chunk_labels_data = chunks[chunk_idx]
            client = clients[client_idx]

            insert_data = []
            for i in range(len(chunk_embeddings)):
                index_data = {"index": {"_index": self.index_name, self.id_col_name: chunk_metadata[i]}}
                if self.with_scalar_labels and self.case_config.use_routing and chunk_labels_data is not None:
                    index_data["routing"] = chunk_labels_data[i]
                insert_data.append(index_data)

                other_data = {self.vector_col_name: chunk_embeddings[i]}
                if self.with_scalar_labels and chunk_labels_data is not None:
                    other_data[self.label_col_name] = chunk_labels_data[i]
                insert_data.append(other_data)

            return self._execute_bulk_with_retries(client, insert_data, f"client {client_idx}")

        results = []
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = []

            for chunk_idx in range(len(chunks)):
                client_idx = chunk_idx % len(clients)
                futures.append(executor.submit(insert_chunk, client_idx, chunk_idx))

            for future in concurrent.futures.as_completed(futures):
                count, error = future.result()
                results.append((count, error))

        from contextlib import suppress

        for client in clients:
            with suppress(Exception):
                client.close()

        total_count = sum(count for count, _ in results)
        errors = [error for _, error in results if error is not None]

        if errors:
            log.warning("Some clients failed to insert data, retrying with single client")
            time.sleep(10)
            return self._insert_with_single_client(embeddings, metadata, labels_data)

        resp = self.client.indices.stats(index=self.index_name)
        log.info(
            f"""Total document count in index after parallel insertion:
                {resp['_all']['primaries']['indexing']['index_total']}""",
        )

        return (total_count, None)

    def _update_ef_search_before_search(self, client: OpenSearch):
        ef_search_value = self.case_config.ef_search
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
                log.info(f"Updating ef_search before search from {current_ef_search} to {ef_search_value}")
                settings_body = {"index": {"knn.algo_param.ef_search": ef_search_value}}
                client.indices.put_settings(index=self.index_name, body=settings_body)
                log.info(f"Successfully updated ef_search to {ef_search_value} before search")

            log.info(f"Current engine: {self.case_config.engine}")
            log.info(f"Current metric_type: {self.case_config.metric_type_name}")

        except Exception as e:
            log.warning(f"Failed to update ef_search parameter before search: {e}")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.

        Returns:
            list[int]: list of k most similar ids to the query embedding.
        """
        assert self.client is not None, "should self.init() first"

        # Configure query based on engine type
        if self.case_config.engine == AWSOS_Engine.s3vector:
            # For s3vector engine, use simplified query without method_parameters
            knn_query = {
                "vector": query,
                "k": k,
                **({"filter": self.filter} if self.filter else {}),
            }
            log.debug("Using simplified knn query for s3vector engine (no method_parameters)")
        else:
            # For other engines (faiss, lucene), use standard query with method_parameters
            knn_query = {
                "vector": query,
                "k": k,
                "method_parameters": self.case_config.search_param(),
                **({"filter": self.filter} if self.filter else {}),
                "rescore": {"oversample_factor": self.case_config.oversample_factor}
                # if self.case_config.use_quant
                # else {}
                ,
            }
            log.debug("Using standard knn query with method_parameters for non-s3vector engines")

        body = {
            "size": k,
            "query": {"knn": {self.vector_col_name: knn_query}},
        }

        try:
            if self._is_serverless:
                resp = self.client.search(
                    index=self.index_name,
                    body=body,
                    size=k,
                    _source=["id"],
                    preference="_only_local" if self.case_config.number_of_shards == 1 else None,
                    routing=self.routing_key,
                )
                try:
                    return [int(h["_source"]["id"]) for h in resp["hits"]["hits"]]
                except Exception:
                    return []
            else:
                resp = self.client.search(
                    index=self.index_name,
                    body=body,
                    size=k,
                    _source=False,
                    docvalue_fields=[self.id_col_name],
                    stored_fields="_none_",
                    preference="_only_local" if self.case_config.number_of_shards == 1 else None,
                    routing=self.routing_key,
                )
                log.debug(f"Search took: {resp['took']}")
                log.debug(f"Search shards: {resp['_shards']}")
                log.debug(f"Search hits total: {resp['hits']['total']}")
                try:
                    return [int(h["fields"][self.id_col_name][0]) for h in resp["hits"]["hits"]]
                except Exception:
                    return []
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise e from None

    def prepare_filter(self, filters: Filter):
        self.routing_key = None
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            # Serverless stores the benchmark id in the "id" field of _source since it
            # does not support custom _id. Filter on that stored field instead of _id.
            filter_field = "id" if self._is_serverless else self.id_col_name
            self.filter = {"range": {filter_field: {"gt": filters.int_value}}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {"term": {self.label_col_name: filters.label_value}}
            if self.case_config.use_routing:
                self.routing_key = filters.label_value
        else:
            msg = f"Not support Filter for OpenSearch - {filters}"
            raise ValueError(msg)

    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases."""
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
        ef_search_value = (
            self.case_config.ef_search if self.case_config.ef_search is not None else self.case_config.efSearch
        )
        log.info(f"Updating ef_search parameter to: {ef_search_value}")

        settings_body = {"index": {"knn.algo_param.ef_search": ef_search_value}}
        try:
            self.client.indices.put_settings(index=self.index_name, body=settings_body)
            log.info(f"Successfully updated ef_search to {ef_search_value}")
            log.info(f"Current engine: {self.case_config.engine}")
            log.info(f"Current metric_type: {self.case_config.metric_type}")
        except Exception as e:
            log.warning(f"Failed to update ef_search parameter: {e}")

    def _update_replicas(self):
        if self._is_serverless:
            log.info("Skipping replica updates for OpenSearch Serverless")
            return

        index_settings = self.client.indices.get_settings(index=self.index_name)
        current_number_of_replicas = int(index_settings[self.index_name]["settings"]["index"]["number_of_replicas"])
        log.info(
            f"Current Number of replicas are {current_number_of_replicas}"
            f" and changing the replicas to {self.case_config.number_of_replicas}"
        )
        settings_body = {"index": {"number_of_replicas": self.case_config.number_of_replicas}}
        self.client.indices.put_settings(index=self.index_name, body=settings_body)
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
        if self._is_serverless:
            log.info("Skipping manual refresh for OpenSearch Serverless, waiting for auto-refresh...")
            time.sleep(10)
            return

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
        if self._is_serverless:
            log.info("Skipping force merge for OpenSearch Serverless")
            return

        log.info(f"Updating the Index thread qty to {self.case_config.index_thread_qty_during_force_merge}.")

        cluster_settings_body = {
            "persistent": {"knn.algo_param.index_thread_qty": self.case_config.index_thread_qty_during_force_merge}
        }
        self.client.cluster.put_settings(body=cluster_settings_body)

        log.info("Updating the graph threshold to ensure that during merge we can do graph creation.")
        output = self.client.indices.put_settings(
            index=self.index_name, body={"index.knn.advanced.approximate_threshold": "0"}
        )
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
        if self._is_serverless:
            log.info("Skipping warmup API for OpenSearch Serverless")
            return

        if self.case_config.engine != AWSOS_Engine.lucene:
            log.info("Calling warmup API to load graphs into memory")
            warmup_endpoint = f"/_plugins/_knn/warmup/{self.index_name}"
            client.transport.perform_request("GET", warmup_endpoint)
