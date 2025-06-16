import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from elasticsearch.helpers import bulk

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import ElasticCloudIndexConfig

for logger in ("elasticsearch", "elastic_transport"):
    logging.getLogger(logger).setLevel(logging.WARNING)

log = logging.getLogger(__name__)


SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC = 30


class ElasticCloud(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ElasticCloudIndexConfig,
        indice: str = "vdb_bench_indice",  # must be lowercase
        id_col_name: str = "id",
        label_col_name: str = "label",
        vector_col_name: str = "vector",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.indice = indice
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_col_name = vector_col_name
        self.with_scalar_labels = with_scalar_labels

        from elasticsearch import Elasticsearch

        client = Elasticsearch(**self.db_config)

        if drop_old:
            log.info(f"Elasticsearch client drop_old indices: {self.indice}")
            is_existed_res = client.indices.exists(index=self.indice)
            if is_existed_res.raw:
                client.indices.delete(index=self.indice)
            self._create_indice(client)

    @contextmanager
    def init(self) -> None:
        """connect to elasticsearch"""
        from elasticsearch import Elasticsearch

        self.client = Elasticsearch(**self.db_config, request_timeout=180)

        yield
        self.client = None
        del self.client

    def _create_indice(self, client: any) -> None:
        mappings = {
            "_source": {"excludes": [self.vector_col_name]},
            "properties": {
                self.id_col_name: {"type": "integer", "store": True},
                self.vector_col_name: {
                    "dims": self.dim,
                    **self.case_config.index_param(),
                },
            },
        }
        settings = {
            "index": {
                "number_of_shards": self.case_config.number_of_shards,
                "number_of_replicas": self.case_config.number_of_replicas,
                "refresh_interval": self.case_config.refresh_interval,
                "merge.scheduler.max_thread_count": self.case_config.merge_max_thread_count,
            }
        }

        try:
            client.indices.create(index=self.indice, mappings=mappings, settings=settings)
        except Exception as e:
            log.warning(f"Failed to create indice: {self.indice} error: {e!s}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert the embeddings to the elasticsearch."""
        assert self.client is not None, "should self.init() first"

        insert_data = (
            [
                (
                    {
                        "_index": self.indice,
                        "_source": {
                            self.id_col_name: metadata[i],
                            self.label_col_name: labels_data[i],
                            self.vector_col_name: embeddings[i],
                        },
                        "_routing": labels_data[i],
                    }
                    if self.case_config.use_routing
                    else {
                        "_index": self.indice,
                        "_source": {
                            self.id_col_name: metadata[i],
                            self.label_col_name: labels_data[i],
                            self.vector_col_name: embeddings[i],
                        },
                    }
                )
                for i in range(len(embeddings))
            ]
            if self.with_scalar_labels
            else [
                {
                    "_index": self.indice,
                    "_source": {
                        self.id_col_name: metadata[i],
                        self.vector_col_name: embeddings[i],
                    },
                }
                for i in range(len(embeddings))
            ]
        )
        try:
            bulk_insert_res = bulk(self.client, insert_data)
            return (bulk_insert_res[0], None)
        except Exception as e:
            log.warning(f"Failed to insert data: {self.indice} error: {e!s}")
            return (0, e)

    def prepare_filter(self, filters: Filter):
        self.routing_key = None
        if filters.type == FilterOp.NonFilter:
            self.filter = []
        elif filters.type == FilterOp.NumGE:
            self.filter = {"range": {self.id_col_name: {"gt": filters.int_value}}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {"term": {self.label_col_name: filters.label_value}}
            if self.case_config.use_routing:
                self.routing_key = filters.label_value
        else:
            msg = f"Not support Filter for Milvus - {filters}"
            raise ValueError(msg)

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
            list[tuple[int, float]]: list of k most similar embeddings in (id, score) tuple to the query embedding.
        """
        assert self.client is not None, "should self.init() first"

        if self.case_config.use_rescore:
            oversample_k = int(k * self.case_config.oversample_ratio)
            oversample_num_candidates = int(self.case_config.num_candidates * self.case_config.oversample_ratio)
            knn = {
                "field": self.vector_col_name,
                "k": oversample_k,
                "num_candidates": oversample_num_candidates,
                "filter": self.filter,
                "query_vector": query,
            }
            rescore = {
                "window_size": oversample_k,
                "query": {
                    "rescore_query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"cosineSimilarity(params.queryVector, '{self.vector_col_name}')",
                                "params": {"queryVector": query},
                            },
                        }
                    },
                    "query_weight": 0,
                    "rescore_query_weight": 1,
                },
            }
        else:
            knn = {
                "field": self.vector_col_name,
                "k": k,
                "num_candidates": self.case_config.num_candidates,
                "filter": self.filter,
                "query_vector": query,
            }
            rescore = None
        size = k

        res = self.client.search(
            index=self.indice,
            knn=knn,
            routing=self.routing_key,
            rescore=rescore,
            size=size,
            _source=False,
            docvalue_fields=[self.id_col_name],
            stored_fields="_none_",
            filter_path=[f"hits.hits.fields.{self.id_col_name}"],
        )
        return [h["fields"][self.id_col_name][0] for h in res["hits"]["hits"]]

    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases."""
        assert self.client is not None, "should self.init() first"
        self.client.indices.refresh(index=self.indice)
        if self.case_config.use_force_merge:
            force_merge_task_id = self.client.indices.forcemerge(
                index=self.indice,
                max_num_segments=1,
                wait_for_completion=False,
            )["task"]
            log.info(f"Elasticsearch force merge task id: {force_merge_task_id}")
            while True:
                time.sleep(SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC)
                task_status = self.client.tasks.get(task_id=force_merge_task_id)
                if task_status["completed"]:
                    return
