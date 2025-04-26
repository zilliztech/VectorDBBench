from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum

from pydantic import BaseModel, SecretStr, validator


class MetricType(str, Enum):
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"
    DP = "DP"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"


class IndexType(str, Enum):
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_PQ = "HNSW_PQ"
    HNSW_PRQ = "HNSW_PRQ"
    DISKANN = "DISKANN"
    STREAMING_DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    IVFPQ = "IVF_PQ"
    IVFSQ8 = "IVF_SQ8"
    IVF_RABITQ = "IVF_RABITQ"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    ES_HNSW = "hnsw"
    ES_IVFFlat = "ivfflat"
    GPU_IVF_FLAT = "GPU_IVF_FLAT"
    GPU_BRUTE_FORCE = "GPU_BRUTE_FORCE"
    GPU_IVF_PQ = "GPU_IVF_PQ"
    GPU_CAGRA = "GPU_CAGRA"
    SCANN = "scann"
    NONE = "NONE"


class SQType(str, Enum):
    SQ6 = "SQ6"
    SQ8 = "SQ8"
    BF16 = "BF16"
    FP16 = "FP16"
    FP32 = "FP32"


class DBConfig(ABC, BaseModel):
    """DBConfig contains the connection info of vector database

    Args:
        db_label(str): label to distinguish different types of DB of the same database.

            MilvusConfig.db_label = 2c8g
            MilvusConfig.db_label = 16c64g
            ZillizCloudConfig.db_label = 1cu-perf
    """

    db_label: str = ""
    version: str = ""
    note: str = ""

    @staticmethod
    def common_short_configs() -> list[str]:
        """
        short input, such as `db_label`, `version`
        """
        return ["version", "db_label"]

    @staticmethod
    def common_long_configs() -> list[str]:
        """
        long input, such as `note`
        """
        return ["note"]

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if field.name in cls.common_short_configs() or field.name in cls.common_long_configs():
            return v
        if not v and isinstance(v, str | SecretStr):
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

    @abstractmethod
    @contextmanager
    def init(self) -> None:
        """create and destory connections to database.
        Why contextmanager:

            In multiprocessing search tasks, vectordbbench might init
            totally hundreds of thousands of connections with DB server.

            Too many connections may drain local FDs or server connection resources.
            If the DB client doesn't have `close()` method, just set the object to None.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        raise NotImplementedError

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        return False

    @abstractmethod
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
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

    @abstractmethod
    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(optimize) will be recorded as "load_duration" metric
        Optimize's execution time is limited, the limited time is based on cases.
        """
        raise NotImplementedError
