from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Type
from contextlib import contextmanager

from pydantic import BaseModel, validator, SecretStr


class MetricType(str, Enum):
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"


class IndexType(str, Enum):
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    ES_HNSW = "hnsw"


class DBConfig(ABC, BaseModel):
    """DBConfig contains the connection info of vector database

    Args:
        db_label(str): label to distinguish different types of DB of the same database.

            MilvusConfig.db_label = 2c8g
            MilvusConfig.db_label = 16c64g
            ZillizCloudConfig.db_label = 1cu-perf
    """

    db_label: str = ""

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @validator("*")
    def not_empty_field(cls, v, field):
        if field.name == "db_label":
            return v
        if isinstance(v, (str, SecretStr)) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class DBCaseConfig(ABC):
    """Case specific vector database configs, usually uesed for index params like HNSW"""
    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError


class EmptyDBCaseConfig(BaseModel, DBCaseConfig):
    """EmptyDBCaseConfig will be used if the vector database has no case specific configs"""
    null: str | None = None
    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}


class VectorDB(ABC):
    """Each VectorDB will be __init__ once for one case, the object will be copied into multiple processes.

    In each process, the benchmark cases ensure VectorDB.init() calls before any other methods operations

    insert_embeddings, search_embedding, and, optimize will be timed for each call.

    Examples:
        >>> milvus = Milvus()
        >>> with milvus.init():
        >>>     milvus.insert_embeddings()
        >>>     milvus.search_embedding()
    """

    @abstractmethod
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str,
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        """Initialize wrapper around the vector database client.

        Please drop the existing collection if drop_old is True. And create collection
        if collection not in the Vector Database

        Args:
            dim(int): the dimension of the dataset
            db_config(dict): configs to establish connections with the vector database
            db_case_config(DBCaseConfig | None): case specific configs for indexing and searching
            drop_old(bool): whether to drop the existing collection of the dataset.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def config_cls(self) -> Type[DBConfig]:
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def case_config_cls(self, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        raise NotImplementedError


    @abstractmethod
    @contextmanager
    def init(self) -> None:
        """ create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        raise NotImplementedError

    @abstractmethod
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        """Insert the embeddings to the vector database. The default number of embeddings for
        each insert_embeddings is 5000.

        Args:
            embeddings(list[list[float]]): list of embedding to add to the vector database.
            metadatas(list[int]): metadata associated with the embeddings, for filtering.
            **kwargs(Any): vector database specific parameters.

        Returns:
            int: inserted data count
        """
        raise NotImplementedError

    @abstractmethod
    def search_embedding(
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

    # TODO: remove
    @abstractmethod
    def optimize(self):
        """optimize will be called between insertion and search in performance cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(optimize) will be recorded as "load_duration" metric
        Optimize's execution time is limited, the limited time is based on cases.
        """
        raise NotImplementedError

    # TODO: remove
    @abstractmethod
    def ready_to_load(self):
        """ready_to_load will be called before load in load cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy load cases.
        """
        raise NotImplementedError
