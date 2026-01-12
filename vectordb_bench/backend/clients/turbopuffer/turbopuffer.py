"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager

import turbopuffer as tpuf

from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB

log = logging.getLogger(__name__)


class TurboPuffer(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TurboPufferIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.api_key = db_config.get("api_key", "")
        self.api_base_url = db_config.get("api_base_url", "")
        self.namespace = db_config.get("namespace", "")
        self.db_case_config = db_case_config
        self.metric = db_case_config.parse_metric()

        self._vector_field = "vector"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.with_scalar_labels = with_scalar_labels

        # Initialize client with new SDK pattern
        self.client = tpuf.Turbopuffer(api_key=self.api_key, base_url=self.api_base_url)

        if drop_old:
            log.info(f"Drop old. delete the namespace: {self.namespace}")
            ns = self.client.namespace(self.namespace)
            try:
                ns.delete_all()
            except Exception as e:
                log.warning(f"Failed to delete all. Error: {e}")

    @contextmanager
    def init(self):
        self.ns = self.client.namespace(self.namespace)
        yield

    def optimize(self, data_size: int | None = None):
        # turbopuffer responds to the request
        #   once the cache warming operation has been started.
        # It does not wait for the operation to complete,
        #   which can take multiple minutes for large namespaces.
        self.ns.hint_cache_warm()
        log.info(f"warming up but no api waiting for complete. just sleep {self.db_case_config.time_wait_warmup}s")
        time.sleep(self.db_case_config.time_wait_warmup)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        try:
            if self.with_scalar_labels:
                self.ns.write(
                    columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: embeddings,
                        self._scalar_label_field: labels_data,
                    },
                    distance_metric=self.metric,
                )
            else:
                self.ns.write(
                    columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: embeddings,
                    },
                    distance_metric=self.metric,
                )
        except Exception as e:
            log.warning(f"Failed to insert. Error: {e}")
        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        res = self.ns.query(
            rank_by=("vector", "ANN", query),
            top_k=k,
            filters=self.expr,
        )
        return [row.id for row in res.rows] if res.rows is not None else []

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            self.expr = (self._scalar_id_field, "Gte", filters.int_value)
        elif filters.type == FilterOp.StrEqual:
            self.expr = (self._scalar_label_field, "Eq", filters.label_value)
        else:
            msg = f"Not support Filter for TurboPuffer - {filters}"
            raise ValueError(msg)
