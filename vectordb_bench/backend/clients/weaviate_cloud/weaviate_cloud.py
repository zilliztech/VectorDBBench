"""Wrapper around the Weaviate vector database over VectorDB"""

import logging
from collections.abc import Iterable
from contextlib import contextmanager

import weaviate
from weaviate.exceptions import WeaviateBaseError

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)


class WeaviateCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the weaviate vector database."""
        db_config.update(
            {
                "auth_client_secret": weaviate.AuthApiKey(
                    api_key=db_config.get("auth_client_secret"),
                ),
            },
        )
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        self._scalar_field = "key"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        from weaviate import Client

        client = Client(**db_config)
        if drop_old:
            try:
                if client.schema.exists(self.collection_name):
                    log.info(f"weaviate client drop_old collection: {self.collection_name}")
                    client.schema.delete_class(self.collection_name)
            except WeaviateBaseError as e:
                log.warning(f"Failed to drop collection: {self.collection_name} error: {e!s}")
                raise e from None
        self._create_collection(client)
        client = None

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        from weaviate import Client

        self.client = Client(**self.db_config)
        yield
        self.client = None
        del self.client

    def optimize(self, data_size: int | None = None):
        assert self.client.schema.exists(self.collection_name)
        self.client.schema.update_config(
            self.collection_name,
            {"vectorIndexConfig": self.case_config.search_param()},
        )

    def _create_collection(self, client: weaviate.Client) -> None:
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
                ],
            }
            class_obj["vectorIndexConfig"] = self.case_config.index_param()
            try:
                client.schema.create_class(class_obj)
            except WeaviateBaseError as e:
                log.warning(f"Failed to create collection: {self.collection_name} error: {e!s}")
                raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into Weaviate"""
        assert self.client.schema.exists(self.collection_name)
        insert_count = 0
        try:
            with self.client.batch as batch:
                batch.batch_size = len(metadata)
                batch.dynamic = True
                res = []
                for i in range(len(metadata)):
                    res.append(
                        batch.add_data_object(
                            {self._scalar_field: metadata[i]},
                            class_name=self.collection_name,
                            vector=embeddings[i],
                        ),
                    )
                    insert_count += 1
                return (len(res), None)
        except WeaviateBaseError as e:
            log.warning(f"Failed to insert data, error: {e!s}")
            return (insert_count, e)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results with distance.
        Should call self.init() first.
        """
        assert self.client.schema.exists(self.collection_name)

        query_obj = (
            self.client.query.get(self.collection_name, [self._scalar_field])
            .with_additional("distance")
            .with_near_vector({"vector": query})
            .with_limit(k)
        )
        if filters:
            where_filter = {
                "path": "key",
                "operator": "GreaterThanEqual",
                "valueInt": filters.get("id"),
            }
            query_obj = query_obj.with_where(where_filter)

        # Perform the search.
        res = query_obj.do()

        # Organize results.
        return [result[self._scalar_field] for result in res["data"]["Get"][self.collection_name]]
