"""Wrapper around the Weaviate vector database over VectorDB"""

import logging
from typing import Any, Iterable

import weaviate
from weaviate.exceptions import WeaviateBaseError

from .api import VectorDB
from .db_case_config import DBCaseConfig

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

        self._scalar_field = "key"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        if drop_old:
            from weaviate import Client
            client = Client(**db_config)
            if client.schema.exists(self.collection_name):
                client.schema.delete_class(self.collection_name)

        self._create_collection(client)


    def init(self) -> None:
        from weaviate import Client
        self.client = Client(**self.db_config)

    def ready_to_load(self):
        """Should call insert first, do nothing"""
        pass

    def ready_to_search(self):
        assert self.client.schema.exists(self.collection_name)
        self.client.schema.update_config(self.collection_name, {"vectorIndexConfig": self.case_config.search_param() } )

    def _create_collection(self, client):
        if not client.schema.exists(self.collection_name):
            log.info(f"Create collection: {self.collection_name}")
            class_obj = {
                "class": self.collection_name,
                "vectorizer": "none",
                "properties": [
                    {
                        "dataType": ["int"],
                        "name": self._scalar_field,
                    },
                ]
            }
            class_obj["vectorIndexConfig"] = self.case_config.index_param()
            try:
                client.schema.create_class(class_obj)
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
        assert self.client.schema.exists(self.collection_name)

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
        ret = [(result[self._scalar_field], result["_additional"]["distance"]) for result in res["data"]["Get"][self.collection_name]]

        return ret

