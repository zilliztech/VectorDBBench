import logging
import time
from contextlib import contextmanager

from vectordb_bench.backend.filter import Filter, FilterOp

from ..elastic_cloud.elastic_cloud import ElasticCloud
from .config import TencentElasticsearchIndexConfig

for logger in ("elasticsearch", "elastic_transport"):
    logging.getLogger(logger).setLevel(logging.WARNING)

log = logging.getLogger(__name__)


SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC = 30


class TencentElasticsearch(ElasticCloud):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    @contextmanager
    def init(self) -> None:
        """connect to elasticsearch"""
        from elasticsearch import Elasticsearch

        self.client = Elasticsearch(**self.db_config, request_timeout=1800)

        yield
        self.client = None
        del self.client

    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases."""
        assert self.client is not None, "should self.init() first"
        self.client.indices.refresh(index=self.indice)
        time.sleep(SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC)
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
