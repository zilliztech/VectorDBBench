"""Wrapper around the Milvus vector database over VectorDB"""

import logging
from typing import Any, Iterable

from sklearn import preprocessing
from pymilvus import Collection, utility
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusException

from .db_case_config import DBCaseConfig, MetricType
from .api import VectorDB

log = logging.getLogger(__name__)


class Milvus(VectorDB):
    def __init__(
        self,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        if drop_old:
            log.info(f"Milvus client drop_old collection: {self.collection_name}")
            from pymilvus import connections
            connections.connect(**self.db_config, timeout=30)
            utility.drop_collection(self.collection_name)
            self.col = None
            connections.disconnect("default")

        self._primary_field = "pk"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

    def init(self) -> None:
        from pymilvus import connections
        self.col: Collection | None = None

        connections.connect(**self.db_config)
        # Grab the existing colection if it exists
        if utility.has_collection(self.collection_name):
            self.col = Collection(self.collection_name)

    def ready_to_load(self):
        assert self.col, "Please call self.init() before"
        if not self.col.has_index(index_name=self._index_name):
            log.info("Milvus create index and load")
            try:
                # this is sync
                self.col.create_index(
                    self._vector_field,
                    self.case_config.index_param(),
                    index_name=self._index_name,
                )

                # this is also sync
                self.col.load()
            except Exception as e:
                log.warning(f"Milvus ready to load error: {e}")
                raise e from None


    def ready_to_search(self):
        assert self.col, "Please call self.init() before"
        if not self.col.has_index(index_name=self._index_name):
            log.info("Milvus flush, compact, create index and load")
            try:
                self.col.flush()
                self.col.compact()
                self.col.wait_for_compaction_completed()

                # this is sync
                self.col.create_index(
                    self._vector_field,
                    self.case_config.index_param(),
                    index_name=self._index_name,
                    #  timeout=600,
                )

                # this is also sync
                self.col.load()
            except Exception as e:
                log.warning(f"Milvus ready to search error: {e}")
                raise e from None

    def _create_collection(self, dim: int) -> Collection:
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        fields = [
            FieldSchema(self._primary_field, DataType.INT64, is_primary=True),
            FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)
        ]

        log.info(f"Create collection: {self.collection_name}")

        # Create the collection
        try:
            return Collection(
                name=self.collection_name,
                schema=CollectionSchema(fields),
                consistency_level="Session",
            )
        except MilvusException as e:
            log.warning(f"Failed to create collection: {self.collection_name} error: {str(e)}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> list[str]:
        """Insert embeddings into Milvus. should call self.init() first"""
        # use the first insert_embeddings to init collection
        if not self.col:
            self.col = self._create_collection(len(embeddings[0]))

        if self.case_config.metric_type == MetricType.COSIN:
            embeddings = preprocessing.normalize(embeddings, norm="l2")

        insert_data = [
                metadata,
                embeddings,
        ]

        try:
            res = self.col.insert(insert_data, **kwargs)
            return res.primary_keys
        except MilvusException as e:
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
        assert self.col is not None

        query = [query]
        if self.case_config.metric_type == MetricType.COSIN:
            from sklearn import preprocessing
            query = preprocessing.normalize(query, norm="l2")

        expr = f"{self._primary_field} {filters.get('metadata')}" if filters else ""

        # Perform the search.
        res = self.col.search(
            data=query,
            anns_field=self._vector_field,
            param=self.case_config.search_param(),
            limit=k,
            expr=expr,
            **kwargs,
        )

        # Organize results.
        ret = [(result.id, result.score) for result in res[0]]
        return ret
