"""Wrapper around the Qdrant over VectorDB"""

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Batch,
    CollectionStatus,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    Range,
    SearchParams,
    VectorParams,
)

from ..api import VectorDB
from .config import QdrantLocalIndexConfig

log = logging.getLogger(__name__)

SECONDS_WAITING_FOR_INDEXING_API_CALL = 5
QDRANT_BATCH_SIZE = 100


def qdrant_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    collection_exists = True

    try:
        client.get_collection(collection_name)
    except Exception:
        collection_exists = False

    return collection_exists


class QdrantLocal(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: QdrantLocalIndexConfig,
        collection_name: str = "QdrantLocalCollection",
        drop_old: bool = False,
        name: str = "QdrantLocal",
        **kwargs,
    ):
        """Initialize wrapper around the qdrant."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.search_parameter = self.case_config.search_param()
        self.collection_name = collection_name
        self.client = None

        self._primary_field = "pk"
        self._vector_field = "vector"

        client = QdrantClient(**self.db_config)

        # Lets just print the parameters here for double check
        log.info(f"Case config: {self.case_config.index_param()}")
        log.info(f"Search parameter: {self.search_parameter}")

        if drop_old and qdrant_collection_exists(client, self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            client.delete_collection(self.collection_name)

        if not qdrant_collection_exists(client, self.collection_name):
            log.info(f"{self.name} create collection: {self.collection_name}")
            self._create_collection(dim, client)

        client = None

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        # create connection
        self.client = QdrantClient(**self.db_config)
        yield
        self.client = None
        del self.client

    def _create_collection(self, dim: int, qdrant_client: QdrantClient):
        log.info(f"Create collection: {self.collection_name}")
        log.info(
            f"Index parameters: m={self.case_config.index_param()['m']}, "
            f"ef_construct={self.case_config.index_param()['ef_construct']}, "
            f"on_disk={self.case_config.index_param()['on_disk']}"
        )

        # If the on_disk is true, we enable both on disk index and vectors.
        try:
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=self.case_config.index_param()["distance"],
                    on_disk=self.case_config.index_param()["on_disk"],
                ),
                hnsw_config=HnswConfigDiff(
                    m=self.case_config.index_param()["m"],
                    ef_construct=self.case_config.index_param()["ef_construct"],
                    on_disk=self.case_config.index_param()["on_disk"],
                ),
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

    def optimize(self, data_size: int | None = None):
        assert self.client, "Please call self.init() before"
        # wait for vectors to be fully indexed
        try:
            while True:
                info = self.client.get_collection(self.collection_name)
                time.sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)
                if info.status != CollectionStatus.GREEN:
                    continue
                if info.status == CollectionStatus.GREEN:
                    log.info(f"Finishing building index for collection: {self.collection_name}")
                    msg = (
                        f"Stored vectors: {info.vectors_count}, Indexed vectors: {info.indexed_vectors_count}, "
                        f"Collection status: {info.indexed_vectors_count}"
                    )
                    log.info(msg)
                    return

        except Exception as e:
            log.warning(f"QdrantCloud ready to search error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of embeddings
            metadata(list[int]): list of metadata
            kwargs: other arguments

        Returns:
            tuple[int, Exception]: number of embeddings inserted and exception if any
        """
        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0

        # disable indexing for quick insertion
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        try:
            for offset in range(0, len(embeddings), QDRANT_BATCH_SIZE):
                vectors = embeddings[offset : offset + QDRANT_BATCH_SIZE]
                ids = metadata[offset : offset + QDRANT_BATCH_SIZE]
                payloads = [{self._primary_field: v} for v in ids]
                _ = self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=Batch(ids=ids, payloads=payloads, vectors=vectors),
                )
                insert_count += QDRANT_BATCH_SIZE
            # enable indexing after insertion
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(indexing_threshold=100),
            )

        except Exception as e:
            log.info(f"Failed to insert data, {e}")
            return insert_count, e
        else:
            return insert_count, None

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
        assert self.client is not None

        f = None
        if filters:
            f = Filter(
                must=[
                    FieldCondition(
                        key=self._primary_field,
                        range=Range(
                            gt=filters.get("id"),
                        ),
                    ),
                ],
            )
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            limit=k,
            query_filter=f,
            search_params=SearchParams(**self.search_parameter),
        ).points

        return [result.id for result in res]
