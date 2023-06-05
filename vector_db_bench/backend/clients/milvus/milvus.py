"""Wrapper around the Milvus vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Any, Iterable, Type

from pymilvus import Collection, utility
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusException

from ..api import VectorDB, DBCaseConfig, DBConfig, IndexType
from .config import MilvusConfig, _milvus_case_config


log = logging.getLogger(__name__)


class Milvus(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        name: str = "Milvus",
    ):
        """Initialize wrapper around the milvus vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        self._primary_field = "pk"
        self._scalar_field = "id"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        from pymilvus import connections
        connections.connect(**self.db_config, timeout=30)
        if drop_old and utility.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(self._primary_field, DataType.INT64, is_primary=True),
                FieldSchema(self._scalar_field, DataType.INT64),
                FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)
            ]

            log.info(f"{self.name} create collection: {self.collection_name}")

            # Create the collection
            coll = Collection(
                name=self.collection_name,
                schema=CollectionSchema(fields),
                consistency_level="Session",
            )

            #  self._pre_load(coll)

        connections.disconnect("default")

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return MilvusConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return _milvus_case_config.get(index_type)


    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        from pymilvus import connections
        self.col: Collection | None = None

        connections.connect(**self.db_config, timeout=60)
        # Grab the existing colection with connections
        self.col = Collection(self.collection_name)

        yield
        connections.disconnect("default")

    def _pre_load(self, coll: Collection):
        if not coll.has_index(index_name=self._index_name):
            log.info(f"{self.name} create index and load")
            try:
                coll.create_index(
                    self._vector_field,
                    self.case_config.index_param(),
                    index_name=self._index_name,
                )

                coll.load()
            except Exception as e:
                log.warning(f"{self.name} pre load error: {e}")
                raise e from None

    def _optimize(self):
        log.info(f"{self.name} optimizing before search")
        try:
            self.col.flush()
            self.col.compact()
            self.col.wait_for_compaction_completed()

            # wait for index done and load refresh
            self.col.create_index(
                self._vector_field,
                self.case_config.index_param(),
                index_name=self._index_name,
            )
            #  utility.wait_for_index_building_complete(self.collection_name)
            #  self.col.load(_refresh=True)
            self.col.load()
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def ready_to_load(self):
        assert self.col, "Please call self.init() before"
        self._pre_load(self.col)
        pass

    def ready_to_search(self):
        assert self.col, "Please call self.init() before"
        self._optimize()

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> int:
        """Insert embeddings into Milvus. should call self.init() first"""
        # use the first insert_embeddings to init collection
        assert self.col is not None
        insert_data = [
                metadata,
                metadata,
                embeddings,
        ]

        try:
            res = self.col.insert(insert_data, **kwargs)
            return len(res.primary_keys)
        except MilvusException as e:
            log.warning("Failed to insert data")
            raise e from None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.col is not None

        expr = f"{self._scalar_field} {filters.get('metadata')}" if filters else ""

        # Perform the search.
        res = self.col.search(
            data=[query],
            anns_field=self._vector_field,
            param=self.case_config.search_param(),
            limit=k,
            expr=expr,
        )

        # Organize results.
        ret = [result.id for result in res[0]]
        return ret
