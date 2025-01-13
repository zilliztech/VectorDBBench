import json
import logging
import time
from contextlib import contextmanager

from alibabacloud_ha3engine_vector import client, models
from alibabacloud_ha3engine_vector.models import QueryRequest
from alibabacloud_searchengine20211025 import models as searchengine_models
from alibabacloud_searchengine20211025.client import Client as searchengineClient
from alibabacloud_tea_openapi import models as open_api_models

from ..api import MetricType, VectorDB
from .config import AliyunOpenSearchIndexConfig

log = logging.getLogger(__name__)

ALIYUN_OPENSEARCH_MAX_SIZE_PER_BATCH = 2 * 1024 * 1024  # 2MB
ALIYUN_OPENSEARCH_MAX_NUM_PER_BATCH = 100


class AliyunOpenSearch(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AliyunOpenSearchIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.control_client = None
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.instance_id = db_config["host"].split(".")[0].replace("http://", "").replace("https://", "")

        self._primary_field = "id"
        self._scalar_field = "int_id"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        self.batch_size = int(
            min(
                ALIYUN_OPENSEARCH_MAX_SIZE_PER_BATCH / (dim * 25),
                ALIYUN_OPENSEARCH_MAX_NUM_PER_BATCH,
            ),
        )

        log.info(f"Aliyun_OpenSearch client config: {self.db_config}")
        control_config = open_api_models.Config(
            access_key_id=self.db_config["ak"],
            access_key_secret=self.db_config["sk"],
            endpoint=self.db_config["control_host"],
        )
        self.control_client = searchengineClient(control_config)

        if drop_old:
            log.info(f"aliyun_OpenSearch client drop old index: {self.collection_name}")
            if self._index_exists(self.control_client):
                self._modify_index(self.control_client)
            else:
                self._create_index(self.control_client)

    def _create_index(self, client: searchengineClient):
        create_table_request = searchengine_models.CreateTableRequest()
        create_table_request.name = self.collection_name
        create_table_request.primary_key = self._primary_field
        create_table_request.partition_count = 1
        create_table_request.field_schema = {
            self._primary_field: "INT64",
            self._vector_field: "MULTI_FLOAT",
            self._scalar_field: "INT64",
        }
        vector_index = searchengine_models.ModifyTableRequestVectorIndex()
        vector_index.index_name = self._index_name
        vector_index.dimension = self.dim
        vector_index.distance_type = self.case_config.distance_type()
        vector_index.vector_field = self._vector_field
        vector_index.vector_index_type = "HNSW"

        advance_params = searchengine_models.ModifyTableRequestVectorIndexAdvanceParams()
        str_max_neighbor_count = f'"proxima.hnsw.builder.max_neighbor_count":{self.case_config.M}'
        str_efc = f'"proxima.hnsw.builder.efconstruction":{self.case_config.ef_construction}'
        str_enable_adsampling = '"proxima.hnsw.builder.enable_adsampling":true'
        str_slack_pruning_factor = '"proxima.hnsw.builder.slack_pruning_factor":1.1'
        str_thread_count = '"proxima.hnsw.builder.thread_count":16'

        params = ",".join(
            [
                str_max_neighbor_count,
                str_efc,
                str_enable_adsampling,
                str_slack_pruning_factor,
                str_thread_count,
            ],
        )
        advance_params.build_index_params = params
        advance_params.search_index_params = (
            '{"proxima.hnsw.searcher.ef":400,"proxima.hnsw.searcher.dynamic_termination.prob_threshold":0.7}'
        )
        vector_index.advance_params = advance_params
        create_table_request.vector_index = [vector_index]

        try:
            response = client.create_table(self.instance_id, create_table_request)
            log.info(f"create table success: {response.body}")
        except Exception as error:
            log.info(error.message)
            log.info(error.data.get("Recommend"))
            log.info(f"Failed to create index: error: {error!s}")
            raise error from None

        # check if index create success
        self._active_index(client)

    # check if index create success
    def _active_index(self, client: searchengineClient) -> None:
        retry_times = 0
        while True:
            time.sleep(10)
            log.info(f"begin to {retry_times} times get table")
            retry_times += 1
            response = client.get_table(self.instance_id, self.collection_name)
            if response.body.result.status == "IN_USE":
                log.info(f"{self.collection_name} table begin to use.")
                return

    def _index_exists(self, client: searchengineClient) -> bool:
        try:
            client.get_table(self.instance_id, self.collection_name)
        except Exception as err:
            log.warning(f"get table from searchengine error, err={err}")
            return False
        else:
            return True

    # check if index build success, Insert the embeddings to the vector database after index build success
    def _index_build_success(self, client: searchengineClient) -> None:
        log.info("begin to check if table build success.")
        time.sleep(50)

        retry_times = 0
        while True:
            time.sleep(10)
            log.info(f"begin to {retry_times} times get table fsm")
            retry_times += 1
            request = searchengine_models.ListTasksRequest()
            request.start = (int(time.time()) - 3600) * 1000
            request.end = int(time.time()) * 1000
            response = client.list_tasks(self.instance_id, request)
            fsms = response.body.result
            cur_fsm = None
            for fsm in fsms:
                if fsm["type"] != "datasource_flow_fsm":
                    continue
                if self.collection_name not in fsm["fsmId"]:
                    continue
                cur_fsm = fsm
                break
            if cur_fsm is None:
                log.warning("no build index fsm")
                return
            if cur_fsm["status"] == "success":
                return

    def _modify_index(self, client: searchengineClient) -> None:
        # check if index create success
        self._active_index(client)

        modify_table_request = searchengine_models.ModifyTableRequest()
        modify_table_request.partition_count = 1
        modify_table_request.primary_key = self._primary_field
        modify_table_request.field_schema = {
            self._primary_field: "INT64",
            self._vector_field: "MULTI_FLOAT",
            self._scalar_field: "INT64",
        }
        vector_index = searchengine_models.ModifyTableRequestVectorIndex()
        vector_index.index_name = self._index_name
        vector_index.dimension = self.dim
        vector_index.distance_type = self.case_config.distance_type()
        vector_index.vector_field = self._vector_field
        vector_index.vector_index_type = "HNSW"
        advance_params = searchengine_models.ModifyTableRequestVectorIndexAdvanceParams()

        str_max_neighbor_count = f'"proxima.hnsw.builder.max_neighbor_count":{self.case_config.M}'
        str_efc = f'"proxima.hnsw.builder.efconstruction":{self.case_config.ef_construction}'
        str_enable_adsampling = '"proxima.hnsw.builder.enable_adsampling":true'
        str_slack_pruning_factor = '"proxima.hnsw.builder.slack_pruning_factor":1.1'
        str_thread_count = '"proxima.hnsw.builder.thread_count":16'

        params = ",".join(
            [
                str_max_neighbor_count,
                str_efc,
                str_enable_adsampling,
                str_slack_pruning_factor,
                str_thread_count,
            ],
        )
        advance_params.build_index_params = params
        advance_params.search_index_params = (
            '{"proxima.hnsw.searcher.ef":400,"proxima.hnsw.searcher.dynamic_termination.prob_threshold":0.7}'
        )
        vector_index.advance_params = advance_params

        modify_table_request.vector_index = [vector_index]

        try:
            response = client.modify_table(
                self.instance_id,
                self.collection_name,
                modify_table_request,
            )
            log.info(f"modify table success: {response.body}")
        except Exception as error:
            log.info(error.message)
            log.info(error.data.get("Recommend"))
            log.info(f"Failed to modify index: error: {error!s}")
            raise error from None

        # check if modify index & delete data fsm success
        self._index_build_success(client)

    # get collection records total count
    def _get_total_count(self):
        try:
            response = self.client.stats(self.collection_name)
        except Exception as e:
            log.warning(f"Error querying index: {e}")
        else:
            body = json.loads(response.body)
            log.info(f"stats info: {response.body}")

            if "result" in body and "totalDocCount" in body.get("result"):
                return body.get("result").get("totalDocCount")
            return 0

    @contextmanager
    def init(self) -> None:
        """connect to aliyun opensearch"""
        config = models.Config(
            endpoint=self.db_config["host"],
            protocol="http",
            access_user_name=self.db_config["user"],
            access_pass_word=self.db_config["password"],
        )

        self.client = client.Client(config)

        yield
        self.client = None
        del self.client

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert the embeddings to the opensearch."""
        assert self.client is not None, "should self.init() first"
        assert len(embeddings) == len(metadata)
        insert_count = 0

        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                documents = []
                for i in range(batch_start_offset, batch_end_offset):
                    document_fields = {
                        self._primary_field: metadata[i],
                        self._vector_field: embeddings[i],
                        self._scalar_field: metadata[i],
                        "ops_build_channel": "inc",
                    }
                    document = {"fields": document_fields, "cmd": "add"}
                    documents.append(document)

                push_doc_req = models.PushDocumentsRequest({}, documents)
                self.client.push_documents(
                    self.collection_name,
                    self._primary_field,
                    push_doc_req,
                )
                insert_count += batch_end_offset - batch_start_offset
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
            return (insert_count, e)
        return (insert_count, None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        assert self.client is not None, "should self.init() first"
        search_params = '{"proxima.hnsw.searcher.ef":' + str(self.case_config.ef_search) + "}"

        os_filter = f"{self._scalar_field} {filters.get('metadata')}" if filters else ""

        try:
            request = QueryRequest(
                table_name=self.collection_name,
                vector=query,
                top_k=k,
                search_params=search_params,
                filter=os_filter,
            )
            result = self.client.query(request)
        except Exception as e:
            log.info(f"Error querying index: {e}")
            raise e from e
        else:
            res = json.loads(result.body)
            return [one_res["id"] for one_res in res["result"]]

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        if self.case_config.metric_type == MetricType.COSINE:
            log.info("cosine dataset need normalize.")
            return True

        return False

    def optimize(self, data_size: int):
        log.info(f"optimize count: {data_size}")
        retry_times = 0
        while True:
            time.sleep(10)
            log.info(f"begin to {retry_times} times get optimize table")
            retry_times += 1
            total_count = self._get_total_count()
            # check if the data is inserted
            if total_count == data_size:
                log.info("optimize table finish.")
                return
