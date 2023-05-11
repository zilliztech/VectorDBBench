from abc import ABC, abstractmethod
from typing import Optional, Any, Iterable


class VectorDB(ABC):
    @abstractmethod
    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
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
        **kwargs: Any
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
