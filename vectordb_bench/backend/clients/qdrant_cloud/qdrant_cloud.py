"""Wrapper around the QdrantCloud vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager

from ..api import VectorDB, DBCaseConfig
from qdrant_client.http.models import (
    CollectionStatus,
    VectorParams,
    PayloadSchemaType,
    Batch,
    Filter,
    FieldCondition,
    Range,
)

from qdrant_client import QdrantClient


log = logging.getLogger(__name__)


class QdrantCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "QdrantCloudCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the QdrantCloud vector database."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        self._primary_field = "pk"
        self._vector_field = "vector"

        tmp_client = QdrantClient(**self.db_config)
        if drop_old:
            log.info(f"QdrantCloud client drop_old collection: {self.collection_name}")
            tmp_client.delete_collection(self.collection_name)
            self._create_collection(dim, tmp_client)
        tmp_client = None

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.qdrant_client = QdrantClient(**self.db_config)
        yield
        self.qdrant_client = None
        del(self.qdrant_client)

    def ready_to_load(self):
        pass


    def optimize(self):
        assert self.qdrant_client, "Please call self.init() before"
        # wait for vectors to be fully indexed
        SECONDS_WAITING_FOR_INDEXING_API_CALL = 5
        try:
            while True:
                info = self.qdrant_client.get_collection(self.collection_name)
                time.sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)
                if info.status != CollectionStatus.GREEN:
                    continue
                if info.status == CollectionStatus.GREEN:
                    log.info(f"Stored vectors: {info.vectors_count}, Indexed vectors: {info.indexed_vectors_count}, Collection status: {info.indexed_vectors_count}")
                    return
        except Exception as e:
            log.warning(f"QdrantCloud ready to search error: {e}")
            raise e from None

    def _create_collection(self, dim, qdrant_client: int):
        log.info(f"Create collection: {self.collection_name}")

        try:
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=self.case_config.index_param()["distance"])
            )

            qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name=self._primary_field,
                field_schema=PayloadSchemaType.INTEGER,
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
        **kwargs,
    ) -> (int, Exception):
        """Insert embeddings into Milvus. should call self.init() first"""
        assert self.qdrant_client is not None
        QDRANT_BATCH_SIZE = 500
        try:
            # TODO: counts
            for offset in range(0, len(embeddings), QDRANT_BATCH_SIZE):
                vectors = embeddings[offset: offset + QDRANT_BATCH_SIZE]
                ids = metadata[offset: offset + QDRANT_BATCH_SIZE]
                payloads=[{self._primary_field: v} for v in ids]
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
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results with score.
        Should call self.init() first.
        """
        assert self.qdrant_client is not None

        f = None
        if filters:
            f = Filter(
                must=[FieldCondition(
                    key = self._primary_field,
                    range = Range(
                        gt=filters.get('id'),
                    ),
                )]
            )

        res = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=k,
            query_filter=f,
            #  with_payload=True,
        ),

        ret = [result.id for result in res[0]]
        return ret
