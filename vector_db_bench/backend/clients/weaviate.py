"""Wrapper around the Weaviate vector database over VectorDB"""

import logging
from typing import Any, Iterable
import weaviate
from weaviate.exceptions import WeaviateBaseError
from pydantic import BaseModel

from ...models import (
    IndexType,
    MetricType,
    DBCaseConfig,
)

from .api import VectorDB

log = logging.getLogger(__name__)


class Weaviate(VectorDB):
    def __init__(
        self,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
    ):
        """Initialize wrapper around the weaviate vector database."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.client = weaviate.Client(**db_config)

        if drop_old:
            if self.client.schema.exists(self.collection_name):
                self.client.schema.delete_class(self.collection_name)

        self._scalar_field = "key"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

    def init(self) -> None:
        pass


    def ready_to_search(self):
        assert self.client.schema.exists(self.collection_name)
        self.client.schema.update_config(self.collection_name, {"vectorIndexConfig": self.case_config.search_param() } )

    def _create_collection(self, dim: int):
        if not self.client.schema.exists(self.collection_name):
            log.info(f"Create collection: {self.collection_name}")
            class_obj = {
                "class": self.collection_name,
                "vectorizer": "none",
                "properties": [
                    {
                        "datatype": ["int"],
                        "name": self._scalar_field,
                    },
                ]
            }
            class_obj["vectorIndexConfig"].update(**self.case_config.index_param())
            try:
                self.client.schema.create_class(class_obj)
            except WeaviateBaseError as e:
                log.warning(f"Failed to create collection: {self.collection_name} error: {str(e)}")
                raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> list[str]:
        """Insert embeddings into Weaviate"""
        if not self.client.schema.exists(self.collection_name):
            self._create_collection()
            self.ready_to_search()

        try:
            with self.client.batch as batch:
                batch.batch_size = len(metadata)
                batch.dynamic = True
                res = []
                for i in range(len(metadata)):
                    res.append(batch.add_data_object(
                        {self._scalar_field: metadata[i]},
                        class_name=self.collection_name,
                        vector=embeddings[i]
                    ))
                return res
        except WeaviateBaseError as e:
            log.warning(f"Failed to insert data, error: {str(e)}")
            raise e from None

    def search_embedding_with_score(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Perform a search on a query embedding and return results with distance.
        Should call self.init() first.
        """
        assert self.client.schema.exists(self.collection_name)

        query_obj = self.client.query.get(self.collection_name, [self._scalar_field]).with_additional("distance").with_near_vector({"vector": query}).with_limit(k)
        if filters:
            where_filter = {
                "path": "key",
                "operator": "GreaterThanEqual",
                "valueInt": filters.get('id')
            }
            query_obj = query_obj.with_where(where_filter)


        # Perform the search.
        res = query_obj.do()

        # Organize results.
        ret = [(result["data"]["Get"][self.collection_name][self._scalar_field], result["data"]["Get"][self.collection_name]["_additional"]["distance"]) for result in res]
        return ret

