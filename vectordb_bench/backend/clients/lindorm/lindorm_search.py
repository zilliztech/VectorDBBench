import concurrent.futures
import json
import logging
import time
from contextlib import contextmanager
from typing import Any

from opensearchpy import OpenSearch

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import LindormConfig, LindormIndexConfig

INDEX_BUILD_LEAST_NUM: int = 256

log = logging.getLogger(__name__)

BATCH_SIZE = 500
NUM_WORKERS = 20
IVFPQ_INDEX_BUILD_NUM_CONSTRAINT = 10000


class LindormVector(VectorDB):
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: LindormIndexConfig,
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
        self.index_name = db_config.get("index_name", "vdb_bench")
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.metadata_col_name = "metadata_fields"
        self.writer = None
        self.category_col_names = [f"scalar-{categoryCount}" for categoryCount in [2, 5, 10, 100, 1000]]
        self.category_col_names.append(self.metadata_col_name)
        self.vector_col_name = vector_col_name
        self.with_scalar_labels = with_scalar_labels
        self._pending_post_optimize_sleep = False  # need sleep after optimize
        log.debug(f"Lindorm client config: {self.db_config}")
        log.debug(f"index name: {self.index_name}")
        log.debug(f"Lindorm index config: {self.case_config}")
        client = OpenSearch(**self.db_config)

        if drop_old:
            log.info(f"Lindorm client drop old index: {self.index_name}")
            is_existed = client.indices.exists(index=self.index_name)
            if is_existed:
                client.indices.delete(index=self.index_name)
        if not client.indices.exists(index=self.index_name):
            log.info(f"create index {self.index_name}")
            self._create_index(client)

        self.write_thread_num = 16
        self.write_vector_batch = 1000

    def build_ivfvectorcode_index(self) -> Any:
        log.info("start build  index ...")
        body = {
            "indexName": self.index_name,
            "fieldName": self.vector_col_name,
            "removeOldIndex": "true",
        }

        self.client.transport.perform_request(
            method="POST",
            url="/_plugins/_vector/index/build",
            body=json.dumps(body),
        )

    def wait_index_build_task_finish(self) -> Any:
        log.info("start check ivf vector build task ...")
        body = {
            "indexName": self.index_name,
            "fieldName": self.vector_col_name,
            "taskIds": '["default_' + self.index_name + "_" + self.vector_col_name + '"]',
        }
        max_retries = 1000
        while True:
            max_retries -= 1
            if max_retries < 0:
                raise RuntimeError("check build task terminated because of timeout!")
            response = self.client.transport.perform_request(
                method="GET",
                url="/_plugins/_vector/index/tasks",
                body=json.dumps(body),
            )
            log.info(response)
            if "stage: FINISH" in response.get("payload", [""])[0]:
                break
            time.sleep(10)
        log.info("finish check build task ...")
        return response

    @classmethod
    def config_cls(cls) -> type[LindormConfig]:
        return LindormConfig

    def _create_index(self, client: OpenSearch):
        searchindex_id_pipeline = {"processors": [{"set": {"field": "_searchindex_id", "value": "{{{_id}}}"}}]}
        client.ingest.put_pipeline(id="searchindex_id", body=searchindex_id_pipeline)
        copy_pipeline = {"processors": [{"pipeline": {"name": "searchindex_id"}}]}
        pipeline_name = "copy_id_pipeline"
        client.ingest.put_pipeline(id=pipeline_name, body=copy_pipeline)
        settings = {
            "index": {
                "knn": True,
                "sync.ckp.enabled": False,
                "origin.vector_source_only_includes.enabled": True,
                "knn.query.prefilter_fetch_id_accelerate.enabled": True,
                "default_pipeline": pipeline_name,
                "refresh_interval": "-1",
                "number_of_shards": 4,
                "knn.vector_number_of_regions": self.case_config.vector_number_of_regions or 1,
            }
        }

        index_type = self.case_config.index.value

        if index_type in {IndexType.HNSW, IndexType.IVFPQ, IndexType.IVFBQ}:
            settings["index"]["knn.offline.construction"] = True

        scalar_column = {
            self.metadata_col_name: {
                "type": "integer",
                "meta": (
                    {"_vector_filter": "true"}
                    if index_type in {IndexType.HNSW, IndexType.IVFPQ, IndexType.IVFBQ}
                    else {}
                ),
            }
        }

        label_column = {
            self.label_col_name: {
                "type": "keyword",
                "meta": (
                    {"_vector_filter": "true"}
                    if index_type in {IndexType.HNSW, IndexType.IVFPQ, IndexType.IVFBQ}
                    else {}
                ),
            }
        }

        mappings = {
            "_source": {"excludes": [self.vector_col_name]},  # do not store vector in search index
            "properties": {
                **scalar_column,
                **label_column,
                "_searchindex_id": {
                    "type": "keyword",
                },
                self.vector_col_name: {
                    "type": "knn_vector",
                    "dimension": self.dim,
                    "method": self.case_config.index_param(),
                },
            },
        }
        log.info(f"mappings: {mappings}")
        try:
            client.indices.create(index=self.index_name, body={"settings": settings, "mappings": mappings})
        except Exception as e:
            log.warning(f"Failed to create index: {self.index_name} error: {e!s}")
            raise e from None

    @contextmanager
    def init(self) -> None:
        self.client = OpenSearch(**self.db_config)
        yield
        if self._pending_post_optimize_sleep:
            log.info("Post-optimize sleep 180s (outside of optimize_duration measurement)...")
            time.sleep(180)
            self._pending_post_optimize_sleep = False

    def _prepare_bulk_data(
        self, batch_embeddings: list[list[float]], batch_metadata: list[int], batch_label: list[int] | None
    ) -> list[dict]:
        insert_data = []
        for i in range(len(batch_embeddings)):
            insert_data.append({"index": {"_index": self.index_name, self.id_col_name: batch_metadata[i]}})

            doc_data = {self.vector_col_name: batch_embeddings[i], self.metadata_col_name: batch_metadata[i]}

            if batch_label is not None and i < len(batch_label) and batch_label[i] is not None:
                doc_data[self.label_col_name] = batch_label[i]

            insert_data.append(doc_data)
        return insert_data

    def _insert_batch_func(
        self, batch_embeddings: list[list[float]], batch_metadata: list[int], batch_label: list[int] | None
    ) -> (int, Exception):
        max_retries = 10
        base_wait_seconds = 1
        wait_increment = 5

        for attempt in range(max_retries):
            try:
                insert_data = self._prepare_bulk_data(batch_embeddings, batch_metadata, batch_label)
                self.client.bulk(body=insert_data)
                return (len(batch_embeddings), None)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait_seconds + wait_increment * attempt
                    logging.warning(
                        f"Failed to insert batch data (attempt {attempt + 1}/{max_retries}): "
                        f"{self.index_name} error: {e!s}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logging.exception(f"Failed to insert batch data after {max_retries} attempts: {self.index_name}")
                    return (0, e)
        return (0, RuntimeError("Failed to insert batch data!"))

    def insert_embeddings(
        self, embeddings: list[list[float]], metadata: list[int], labels_data: list[str] | None = None, **kwargs
    ) -> (int, Exception):
        num_threads = self.write_thread_num
        batch_size = self.write_vector_batch

        total_count = len(embeddings)
        batched_embeddings = [embeddings[i : i + batch_size] for i in range(0, total_count, batch_size)]
        batched_metadata = [metadata[i : i + batch_size] for i in range(0, total_count, batch_size)]
        if labels_data is not None:
            batched_labels = [labels_data[i : i + batch_size] for i in range(0, total_count, batch_size)]
        else:
            batched_labels = [None] * len(batched_embeddings)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self._insert_batch_func, batch_emb, batch_meta, batch_label)
                for batch_emb, batch_meta, batch_label in zip(
                    batched_embeddings, batched_metadata, batched_labels, strict=True
                )
            ]

            total_inserted = 0
            final_error = None

            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        count, error = future.result()
                        if error is not None:
                            final_error = error
                            break
                        total_inserted += count
                    except Exception as e:
                        final_error = e
                        break

                if final_error is not None:
                    return (total_inserted, final_error)
                return (total_inserted, None)  # noqa: TRY300

            except Exception as e:
                logging.warning(f"Unexpected error in insert_embeddings: {e!s}")
                return (total_inserted, e)

    def prepare_filter(self, filters: Filter):
        self.lindorm_filter = None

        if filters.type == FilterOp.NonFilter:
            self.lindorm_filter = None
            return
        if filters.type == FilterOp.NumGE:
            self.lindorm_filter = {"range": {self.metadata_col_name: {"gt": filters.int_value}}}
            return
        if filters.type == FilterOp.StrEqual:
            self.lindorm_filter = {"term": {self.label_col_name: filters.label_value}}
            return

    def search_embedding(self, query: list[float], k: int = 100) -> list[int]:
        assert self.client is not None, "should self.init() first"

        query_new = [round(x, 6) for x in query]
        body = {
            "size": k,
            "query": {
                "knn": {
                    self.vector_col_name: {
                        "vector": query_new,
                        "k": k,
                        **({"filter": self.lindorm_filter} if self.lindorm_filter else {}),
                    },
                },
            },
            "ext": self.case_config.search_param(do_filter=(self.lindorm_filter is not None)),
        }
        try:
            resp = self.client.search(index=self.index_name, body=body, size=k, _source=False)
            log.debug(f'Search took: {resp["took"]}')
            log.debug(f'Search shards: {resp["_shards"]}')
            log.debug(f'Search hits total: {resp["hits"]["total"]}')
            if len(resp["hits"]["hits"]) > 0:
                if self.id_col_name in resp["hits"]["hits"][0]:
                    return [int(d["_id"]) for d in resp["hits"]["hits"]]
                if "fields" in resp["hits"]["hits"][0]:
                    return [int(h["fields"][self.id_col_name][0]) for h in resp["hits"]["hits"]]
            else:
                return []
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        try:
            self.client.indices.put_settings(index=self.index_name, body={"index": {"refresh_interval": "15s"}})
            log.info(f"change refresh_interval for {self.index_name} to 15s")
        except Exception as e:
            log.warning(f"Failed to update refresh_interval: {e}")
        self._refresh_index()
        self._force_merge()
        index_type = self.case_config.index.value
        if index_type in {IndexType.HNSW, IndexType.IVFPQ, IndexType.IVFBQ}:
            self.build_ivfvectorcode_index()
            self.wait_index_build_task_finish()
            self._pending_post_optimize_sleep = True
            log.info("Index build completed, will sleep 180s after optimize_duration measurement ends")

    def _refresh_index(self):
        log.debug(f"Starting refresh for index {self.index_name}")
        seconds_waiting_for_refresh_api_call_sec = 30
        while True:
            try:
                log.info("Starting the Refresh Index..")
                self.client.indices.refresh(index=self.index_name)
                break
            except Exception as e:
                log.info(
                    f"Refresh errored out. Sleeping for {seconds_waiting_for_refresh_api_call_sec} sec "
                    f"and then Retrying : {e}"
                )
                time.sleep(seconds_waiting_for_refresh_api_call_sec)
                continue
        log.debug(f"Completed refresh for index {self.index_name}")

    def _force_merge(self):
        log.info("Force Merge...")
        try:
            self.client.indices.forcemerge(index=self.index_name, max_num_segments=1, timeout=1800)
        except Exception:
            log.warning("Force Merge Fail or timeout")

    def ready_to_load(self):
        """ready_to_load will be called before load in load cases."""
