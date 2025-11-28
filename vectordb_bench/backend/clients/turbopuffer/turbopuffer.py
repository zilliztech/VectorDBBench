"""Wrapper around the Turbopuffer vector database over VectorDB"""

import logging
from contextlib import contextmanager

import turbopuffer as tpuf

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)

TURBOPUFFER_MAX_NUM_PER_BATCH = 1000


class Turbopuffer(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the turbopuffer vector database."""
        self.namespace = db_config.get("namespace", "vdbbench")
        self.api_key = db_config.get("api_key", "")
        self.region = db_config.get("region", "us-east-1")
        self.dim = dim
        self.batch_size = TURBOPUFFER_MAX_NUM_PER_BATCH
        self.with_scalar_labels = with_scalar_labels

        # Create temporary client for initialization only
        tmp_client = tpuf.Turbopuffer(api_key=self.api_key, region=self.region)
        tmp_ns = tmp_client.namespace(self.namespace)

        if drop_old:
            try:
                # Delete all vectors in the namespace
                log.info(f"Turbopuffer deleting namespace: {self.namespace}")
                tmp_ns.delete_all()
            except Exception as e:
                log.warning(f"Failed to drop old data: {e}")

        # Don't store the client/namespace to avoid pickling issues
        tmp_client = None
        tmp_ns = None

        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        # Parse distance metric from case config
        if hasattr(db_case_config, 'parse_metric'):
            self.distance_metric = db_case_config.parse_metric()
        else:
            self.distance_metric = "cosine_distance"

    @contextmanager
    def init(self):
        """Create and destroy connections to database."""
        self.client = tpuf.Turbopuffer(api_key=self.api_key, region=self.region)
        self.ns = self.client.namespace(self.namespace)
        yield
        self.ns = None
        self.client = None

    def optimize(self, data_size: int | None = None):
        """Turbopuffer is serverless and auto-optimizes."""
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        assert len(embeddings) == len(metadata)
        insert_count = 0

        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))

                # Prepare batch data
                vectors = []
                for i in range(batch_start_offset, batch_end_offset):
                    row = {
                        'id': metadata[i],  # ID can be int or str
                        'vector': embeddings[i],
                    }
                    if self.with_scalar_labels and labels_data:
                        row[self._scalar_label_field] = labels_data[i]

                    vectors.append(row)

                # Upsert vectors to turbopuffer
                self.ns.write(upsert_rows=vectors, distance_metric=self.distance_metric)
                insert_count += batch_end_offset - batch_start_offset

        except Exception as e:
            log.error(f"Failed to insert embeddings: {e}")
            return insert_count, e

        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Search for k nearest neighbors."""
        try:
            filters = self.expr if hasattr(self, 'expr') and self.expr else None

            result = self.ns.query(
                rank_by=("vector", "ANN", query),
                top_k=k,
                distance_metric=self.distance_metric,
                filters=filters,
            )

            # Extract IDs from results
            return [int(item.id) for item in result.rows] if result.rows else []

        except Exception as e:
            log.error(f"Search failed: {e}")
            return []

    def prepare_filter(self, filters: Filter):
        """Prepare filter expressions for turbopuffer."""
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            # Turbopuffer uses filter format like: ['id', 'Gte', value]
            self.expr = [self._scalar_id_field, 'Gte', filters.int_value]
        elif filters.type == FilterOp.StrEqual:
            # String equality filter
            self.expr = [self._scalar_label_field, 'Eq', filters.label_value]
        else:
            msg = f"Not support Filter for Turbopuffer - {filters}"
            raise ValueError(msg)