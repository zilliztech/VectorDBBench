from abc import ABC, abstractmethod
from typing import Iterable
from contextlib import contextmanager
from .db_case_config import DBCaseConfig


class VectorDB(ABC):
    """Each VectorDB will be __init__ once, the object will be copied into multiple processes.

    In each process, the benchmark cases ensure VectorDB.init() calls before any other methods operations

    insert_embeddings, search_embedding_with_score, and, ready_to_search will be timed for each call.

    Examples:
        db = Milvus()
        with Milvus
    """

    @abstractmethod
    def __init__(
        self,
        dim: int,
        db_config: dict | None,
        db_case_config: DBCaseConfig | None,
        drop_old: bool = False,
        **kwargs
    ) -> None:
        """Init collection"""
        raise NotImplementedError

    @abstractmethod
    @contextmanager
    def init(self) -> None:
        """ connect to DB

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        raise NotImplementedError

    @abstractmethod
    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
    ) -> int:
        """Insert the embeddings to the vector database

        Args:
            embeddings(Iterable[list[float]]): list of embedding to add to the vector database.
            metadatas(list[int], Optional): metadata associated with the embeddings, for filtering
            kwargs(Any): vector database specific parameters.

        Returns:
            int: inserted data count
        """
        raise NotImplementedError

    @abstractmethod
    def search_embedding_with_score(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[int]: list of k most similar embeddings IDs to the query embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def ready_to_search(self):
        """ready_to_search will be called between insertion and search in performance cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(ready_to_search) will be recorded as "ready_elapse" metric
        """
        raise NotImplementedError

    @abstractmethod
    def ready_to_load(self):
        """ready_to_load will be called before load in load cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy load cases.
        """
        raise NotImplementedError
