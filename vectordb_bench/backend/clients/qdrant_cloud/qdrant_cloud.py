"""Wrapper around the QdrantCloud vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Batch,
    CollectionStatus,
    FieldCondition,
    HnswConfigDiff,
    KeywordIndexParams,
    OptimizersConfigDiff,
    PayloadSchemaType,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from qdrant_client.http.models import (
    Filter as QdrantFilter,
)

from vectordb_bench.backend.clients.qdrant_cloud.config import QdrantIndexConfig
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB

log = logging.getLogger(__name__)


SECONDS_WAITING_FOR_INDEXING_API_CALL = 5
QDRANT_BATCH_SIZE = 500


class QdrantCloud(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: QdrantIndexConfig,
        collection_name: str = "QdrantCloudCollection",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the QdrantCloud vector database."""
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.collection_name = collection_name

        self._primary_field = "pk"
        self._scalar_label_field = "label"
        self._vector_field = "vector"

        tmp_client = QdrantClient(**self.db_config)
        self.with_scalar_labels = with_scalar_labels
        if drop_old:
            log.info(f"QdrantCloud client drop_old collection: {self.collection_name}")
            tmp_client.delete_collection(self.collection_name)
            self._create_collection(dim, tmp_client)
        tmp_client = None

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.qdrant_client = QdrantClient(**self.db_config)
        yield
        self.qdrant_client = None
        del self.qdrant_client

    def optimize(self, data_size: int | None = None):
        assert self.qdrant_client, "Please call self.init() before"
        # wait for vectors to be fully indexed
        try:
            while True:
                info = self.qdrant_client.get_collection(self.collection_name)
                time.sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)
                if info.status != CollectionStatus.GREEN:
                    continue
                if info.status == CollectionStatus.GREEN:
                    msg = (
                        f"Stored vectors: {info.vectors_count}, Indexed vectors: {info.indexed_vectors_count}, "
                        f"Collection status: {info.status}, Segment counts: {info.segments_count}"
                    )
                    log.info(msg)
                    return
        except Exception as e:
            log.warning(f"QdrantCloud ready to search error: {e}")
            raise e from None

    def _create_collection(self, dim: int, qdrant_client: QdrantClient):
        log.info(f"Create collection: {self.collection_name}")

        try:
            # whether to use quant (SQ8)
            quantization_config = None
            if self.db_case_config.use_scalar_quant:
                quantization_config = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=self.db_case_config.sq_quantile,
                        always_ram=True,
                    )
                )

            # create collection
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=self.db_case_config.parse_metric(),
                ),
                hnsw_config=HnswConfigDiff(m=self.db_case_config.m, payload_m=self.db_case_config.payload_m),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=self.db_case_config.default_segment_number
                ),
                quantization_config=quantization_config,
            )

            # create payload_index for int-field
            if self.db_case_config.create_payload_int_index:
                qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=self._primary_field,
                    field_schema=PayloadSchemaType.INTEGER,
                )

            # create payload_index for str-field
            if self.with_scalar_labels and self.db_case_config.create_payload_keyword_index:
                qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=self._scalar_label_field,
                    field_schema=KeywordIndexParams(
                        type=PayloadSchemaType.KEYWORD, is_tenant=self.db_case_config.is_tenant
                    ),
                )

        except Exception as e:
            if "already exists!" in str(e):
                return
            log.warning(f"Failed to create collection: {self.collection_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into Milvus. should call self.init() first"""
        assert self.qdrant_client is not None
        try:
            for offset in range(0, len(embeddings), QDRANT_BATCH_SIZE):
                vectors = embeddings[offset : offset + QDRANT_BATCH_SIZE]
                ids = metadata[offset : offset + QDRANT_BATCH_SIZE]
                if self.with_scalar_labels:
                    labels = labels_data[offset : offset + QDRANT_BATCH_SIZE]
                    payloads = [
                        {self._primary_field: pk, self._scalar_label_field: labels[i]} for i, pk in enumerate(ids)
                    ]
                else:
                    payloads = [{self._primary_field: pk} for i, pk in enumerate(ids)]
                _ = self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=Batch(ids=ids, payloads=payloads, vectors=vectors),
                )
        except Exception as e:
            log.info(f"Failed to insert data, {e}")
            return 0, e
        else:
            return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        """Perform a search on a query embedding and return results with score.
        Should call self.init() first.
        """
        assert self.qdrant_client is not None

        res = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=k,
            query_filter=self.query_filter,
            search_params=self.db_case_config.search_param(),
            with_payload=self.db_case_config.with_payload,
        )

        return [r.id for r in res]

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.query_filter = None
        elif filters.type == FilterOp.NumGE:
            self.query_filter = QdrantFilter(
                must=[
                    FieldCondition(
                        key=self._primary_field,
                        range=Range(gte=filters.int_value),
                    ),
                ]
            )
        elif filters.type == FilterOp.StrEqual:
            self.query_filter = QdrantFilter(
                must=[
                    FieldCondition(
                        key=self._scalar_label_field,
                        match={"value": filters.label_value},
                    ),
                ]
            )
        else:
            msg = f"Not support Filter for Qdrant - {filters}"
            raise ValueError(msg)
