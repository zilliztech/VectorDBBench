from abc import ABC, abstractmethod
from typing import Any, Iterable
from . import DBCaseConfig


class VectorDB(ABC):
    def init(db_config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadatas: list[dict] | None = None,
        db_case_config: DBCaseConfig | None = None,
    ) -> list[str]:
        """Insert the embeddings to the vector database

        Args:
            embeddings(Iterable[list[float]]): list of embedding to add to the vector database.
            metadatas(list[dict], Optional): metadatas associated with the embeddings.
            kwargs(Any): vector database specific parameters.

        Returns:
            list[str]: ids from insertion of the embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def search_embedding_with_score(
        self,
        query: list[float],
        k: int = 100,
        filters: Any | None = None,
        db_case_config: DBCaseConfig | None = None,
    ) -> list[tuple[int, float]]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(Any, optional): filtering expression to filter the data while searching.
            kwargs(Any): vector database specific parameters.

        Returns:
            list[tuple[int, float]]: list of k most similar embeddings in (id, score) tuple to the query embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def ready_to_search(self):
        """ready_to_search will be called between insertion and search.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(ready_to_search) will be recorded as "ready_elapse" metric
        """
        raise NotImplementedError
