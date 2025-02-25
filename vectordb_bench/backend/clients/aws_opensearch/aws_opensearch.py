import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from opensearchpy import OpenSearch

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AWSOpenSearchIndexConfig, AWSOS_Engine

log = logging.getLogger(__name__)

WAITING_FOR_REFRESH_SEC = 30
WAITING_FOR_FORCE_MERGE_SEC = 30
SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC = 30


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
        ef_search_value = (
            self.case_config.ef_search if self.case_config.ef_search is not None else self.case_config.efSearch
        )
        log.info(f"Creating index with ef_search: {ef_search_value}")
        log.info(f"Creating index with number_of_replicas: {self.case_config.number_of_replicas}")

        log.info(f"Creating index with engine: {self.case_config.engine}")
        log.info(f"Creating index with metric type: {self.case_config.metric_type_name}")
        log.info(f"All case_config parameters: {self.case_config.__dict__}")

        cluster_settings_body = {
            "persistent": {
                "knn.algo_param.index_thread_qty": self.case_config.index_thread_qty,
                "knn.memory.circuit_breaker.limit": self.case_config.cb_threshold,
            }
        }
        client.cluster.put_settings(cluster_settings_body)
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
        mappings = {
            "_source": {"excludes": [self.vector_col_name], "recovery_source_excludes": [self.vector_col_name]},
            "properties": {
                self.id_col_name: {"type": "integer", "store": True},
                self.label_col_name: {"type": "keyword"},
                self.vector_col_name: {
                    "type": "knn_vector",
                    "dimension": self.dim,
                    "method": self.case_config.index_param(),
                },
            },
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
        insert_data = []
        for i in range(len(embeddings)):
            index_data = {"index": {"_index": self.index_name, self.id_col_name: metadata[i]}}
            if self.with_scalar_labels and self.case_config.use_routing:
                index_data["routing"] = labels_data[i]
            insert_data.append(index_data)

            other_data = {self.vector_col_name: embeddings[i]}
            if self.with_scalar_labels:
                other_data[self.label_col_name] = labels_data[i]
            insert_data.append(other_data)

        try:
            self.client.bulk(insert_data)
            return len(embeddings), None
        except Exception as e:
            log.warning(f"Failed to insert data: {self.index_name} error: {e!s}")
            time.sleep(10)
            return self._insert_with_single_client(embeddings, metadata)

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
            chunks.append((embeddings_list[i:end], metadata[i:end], labels_data[i:end]))

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
                if self.with_scalar_labels and self.case_config.use_routing:
                    index_data["routing"] = chunk_labels_data[i]
                insert_data.append(index_data)

                other_data = {self.vector_col_name: chunk_embeddings[i]}
                if self.with_scalar_labels:
                    other_data[self.label_col_name] = chunk_labels_data[i]
                insert_data.append(other_data)

            try:
                resp = client.bulk(insert_data)
                log.info(f"Client {client_idx} added {len(resp['items'])} documents")
                return len(chunk_embeddings), None
            except Exception as e:
                log.warning(f"Client {client_idx} failed to insert data: {e!s}")
                return 0, e

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
            return self._insert_with_single_client(embeddings, metadata)

        resp = self.client.indices.stats(self.index_name)
        log.info(
            f"""Total document count in index after parallel insertion:
                {resp['_all']['primaries']['indexing']['index_total']}""",
        )

        return (total_count, None)

    def _update_ef_search_before_search(self, client: OpenSearch):
        ef_search_value = (
            self.case_config.ef_search if self.case_config.ef_search is not None else self.case_config.efSearch
        )

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

        body = {
            "size": k,
            "query": {
                "knn": {
                    self.vector_col_name: {
                        "vector": query,
                        "k": k,
                        "method_parameters": self.case_config.search_param(),
                        **({"filter": self.filter} if self.filter else {}),
                        **(
                            {"rescore": {"oversample_factor": self.case_config.oversample_factor}}
                            if self.case_config.use_quant
                            else {}
                        ),
                    }
                }
            },
        }

        try:
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
                # empty results
                return []
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise e from None

    def prepare_filter(self, filters: Filter):
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

        cluster_settings_body = {
            "persistent": {"knn.algo_param.index_thread_qty": self.case_config.index_thread_qty_during_force_merge}
        }
        self.client.cluster.put_settings(cluster_settings_body)

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
        if self.case_config.engine != AWSOS_Engine.lucene:
            log.info("Calling warmup API to load graphs into memory")
            warmup_endpoint = f"/_plugins/_knn/warmup/{self.index_name}"
            client.transport.perform_request("GET", warmup_endpoint)
