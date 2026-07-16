from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, model_validator

from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile


class MetricType(StrEnum):
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"
    DP = "DP"
    BM25 = "BM25"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"


class IndexType(StrEnum):
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_BQ = "HNSW_BQ"
    HNSW_PQ = "HNSW_PQ"
    HNSW_PRQ = "HNSW_PRQ"
    DISKANN = "DISKANN"
    STREAMING_DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    IVFPQ = "IVF_PQ"
    IVFBQ = "IVF_BQ"
    IVFSQ8 = "IVF_SQ8"
    IVF_RABITQ = "IVF_RABITQ"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    FTS = "FTS"
    ES_HNSW = "hnsw"
    ES_HNSW_INT8 = "int8_hnsw"
    ES_HNSW_INT4 = "int4_hnsw"
    ES_HNSW_BBQ = "bbq_hnsw"
    TES_VSEARCH = "vsearch"
    ES_IVFFlat = "ivfflat"
    GPU_IVF_FLAT = "GPU_IVF_FLAT"
    GPU_BRUTE_FORCE = "GPU_BRUTE_FORCE"
    GPU_IVF_PQ = "GPU_IVF_PQ"
    GPU_CAGRA = "GPU_CAGRA"
    SCANN = "scann"
    VCHORDRQ = "vchordrq"
    VCHORDG = "vchordg"
    SCANN_MILVUS = "SCANN_MILVUS"
    SVS_VAMANA = "SVS_VAMANA"
    SVS_VAMANA_LVQ = "SVS_VAMANA_LVQ"
    SVS_VAMANA_LEANVEC = "SVS_VAMANA_LEANVEC"
    Hologres_HGraph = "HGraph"
    Hologres_Graph = "Graph"
    IVF_HNSW_SQ = "IVF_HNSW_SQ"
    IVF_HNSW_PQ = "IVF_HNSW_PQ"
    NONE = "NONE"


class SQType(StrEnum):
    SQ4U = "SQ4U"
    SQ6 = "SQ6"
    SQ8 = "SQ8"
    BF16 = "BF16"
    FP16 = "FP16"
    FP32 = "FP32"


class NonRetryableInsertError(RuntimeError):
    non_retryable = True


class PartialInsertError(NonRetryableInsertError):
    def __init__(
        self,
        message: str,
        *,
        inserted_count: int,
        successful_tenants: dict[str, int] | None = None,
        failed_tenant: str | None = None,
        failed_tenant_count: int | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.inserted_count = inserted_count
        self.successful_tenants = successful_tenants or {}
        self.failed_tenant = failed_tenant
        self.failed_tenant_count = failed_tenant_count
        self.__cause__ = cause


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

    # Field names subclasses allow to be empty (optional creds, alt-route fields).
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset()

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

    @model_validator(mode="before")
    @classmethod
    def not_empty_field(cls, data: any) -> any:
        if not isinstance(data, dict):
            return data
        skip = set(cls.common_short_configs()) | set(cls.common_long_configs()) | cls._extra_empty_skip
        empty = [k for k, v in data.items() if k not in skip and isinstance(v, str) and not v]
        if empty:
            msg = f"Empty field(s): {', '.join(empty)}"
            raise ValueError(msg)
        return data


class DBCaseConfig(ABC):
    """Case specific vector database configs, usually uesed for index params like HNSW"""

    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError

    def apply_fts_manifest(
        self,
        bm25_params: dict[str, float],
        analyzer_params: dict,
    ) -> tuple["DBCaseConfig", dict]:
        """Apply FTS dataset manifest parameters to this case config.

        Full-text search datasets may provide BM25 and analyzer settings used to
        build the mathematical ground truth. Backends that can reproduce those
        settings should return an updated config with supported parameters
        applied. Unsupported parameters must be reported in the returned metadata
        instead of being silently ignored.

        Args:
            bm25_params(dict[str, float]): BM25 parameters from the dataset
                manifest, such as k1, b, and avgdl.
            analyzer_params(dict): analyzer settings from the dataset manifest.

        Returns:
            tuple[DBCaseConfig, dict]: updated config and a report describing
            applied and unapplied BM25/analyzer parameters.
        """
        return self, {
            "applied_bm25_params": {},
            "unapplied_bm25_params": dict(bm25_params),
            "applied_analyzer_params": {},
            "unapplied_analyzer_params": dict(analyzer_params),
        }


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

    "The filtering types supported by the VectorDB Client, default only non-filter"
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    name: str = ""

    # Whether the client can share a single connection across threads.
    # If False, concurrent runners will deep-copy the instance and call
    # init() per thread instead of sharing the parent connection.
    thread_safe: bool = True

    @classmethod
    def filter_supported(cls, filters: Filter) -> bool:
        """Ensure that the filters are supported before testing filtering cases."""
        return filters.type in cls.supported_filter_types

    def prepare_filter(self, filters: Filter):
        """The vector database is allowed to pre-prepare different filter conditions
        to reduce redundancy during the testing process.

        (All search tests in a case use consistent filtering conditions.)"""
        return

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

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile == PayloadProfile.IDS_ONLY

    def has_text_field(self) -> bool:
        return False

    def supports_document_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        if payload_profile == PayloadProfile.IDS_ONLY:
            return True
        if payload_profile == PayloadProfile.TEXT:
            return self.has_text_field()
        return False

    def poll_insert_readiness(self, expected_count: int) -> dict:
        return {"fully_searchable": True, "fully_indexed": True, "additional_parameters": {}}

    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels

    def supports_multitenant(self) -> bool:
        return False

    def validate_multitenant_schema(self) -> None:
        return None

    @classmethod
    def supports_full_text_search(cls) -> bool:
        """Return whether this client implements the full-text search API.

        Backends that return True must implement insert_documents and
        search_documents for raw text documents.
        """
        return False

    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert raw text documents for full-text search cases.

        Args:
            texts(list[str]): raw text documents to index.
            doc_ids(list[str]): stable document IDs aligned with texts.
            **kwargs(Any): backend or runner specific insert parameters.

        Returns:
            tuple[int, Exception | None]: inserted document count and an optional
            error. Implementations should return the count of successfully
            inserted documents even when reporting a partial failure.
        """
        msg = f"{self.name or self.__class__.__name__} does not support full-text document insert"
        raise NotImplementedError(msg)

    def search_documents(
        self,
        query: str,
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        **kwargs,
    ) -> list[str]:
        """Search full-text documents and return ranked document IDs.

        Args:
            query(str): raw query text.
            k(int): number of nearest documents to return. Defaults to 100.
            payload_profile(PayloadProfile): response payload shape requested by
                the benchmark. The API still returns document IDs only; use
                supports_document_payload_profile to reject unsupported profiles.
            **kwargs(Any): backend or runner specific search parameters.

        Returns:
            list[str]: ranked document IDs for the query.
        """
        msg = f"{self.name or self.__class__.__name__} does not support full-text document search"
        raise NotImplementedError(msg)

    @abstractmethod
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert one task-configured batch of embeddings into the vector database.

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
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
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
