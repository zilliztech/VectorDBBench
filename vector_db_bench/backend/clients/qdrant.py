"""Wrapper around the Qdrant vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterable

from .db_case_config import DBCaseConfig
from .api import VectorDB
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    VectorParams,
    PayloadSchemaType,
    Batch,
    Filter,
    FieldCondition,
    Range,
)

from qdrant_client import QdrantClient


log = logging.getLogger(__name__)


class Qdrant(VectorDB):
    def __init__(
        self,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "QdrantCollection",
        drop_old: bool = False,
    ):
        """Initialize wrapper around the Qdrant vector database."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        if drop_old:
            log.info(f"Qdrant client drop_old collection: {self.collection_name}")
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(**self.db_config)
            qdrant_client.delete_collection(self.collection_name)
            self.qdrant_client = None

        self._primary_field = "pk"
        self._vector_field = "vector"

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding_with_score()
        """
        self.qdrant_client = QdrantClient(**self.db_config)
        yield
        pass

    def ready_to_load(self):
        pass


    def ready_to_search(self):
        assert self.qdrant_client, "Please call self.init() before"
        # wait for vectors to be fully indexed
        SECONDS_WAITING_FOR_INDEXING_API_CALL = 5
        try:
            while True:
                collection_info = self.qdrant_client.get_collection(self._collection_name)
                time.sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)
                if collection_info.status != CollectionStatus.GREEN:
                    continue
                if collection_info.status == CollectionStatus.GREEN:
                    log.info(f"Stored vectors: {collection_info.vectors_count}, Indexed vectors: {collection_info.indexed_vectors_count}, Collection status: {collection_info.indexed_vectors_count}")
                    return
        except Exception as e:
            log.warning(f"Qdrant ready to search error: {e}")
            raise e from None

    def _create_collection(self, dim: int):
        log.info(f"Create collection: {self.collection_name}")

        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.EUCLID)
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name=self._primary_field,
                field_schema=PayloadSchemaType.INTEGER,
            )

        except Exception as e:
            log.warning(f"Failed to create collection: {self.collection_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> list[str]:
        """Insert embeddings into Milvus. should call self.init() first"""
        # use the first insert_embeddings to init collection
        if not self.qdrant_client.get_collection(self.collection_name):
            self._create_collection(len(embeddings[0]))

        try:
            # TODO: counts
            _ = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=Batch(payloads={self._primary_field: v for v in metadata}, vectors=embeddings)
            )

            return len(metadata)
        except Exception as e:
            log.warning("Failed to insert data")
            raise e from None

    def search_embedding_with_score(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
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
            query_vector=[query],
            limit=k,
            query_filter=f,
            #  with_payload=True,
        ),

        ret = [(result.id, result.score) for result in res[0]]
        return ret
