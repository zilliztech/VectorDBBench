from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel


class IndexType(str, Enum):
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    ES_HNSW = "hnsw"


class MetricType(str, Enum):
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"


class ESElementType(str, Enum):
    float = "float"  # 4 byte
    byte = "byte"  # 1 byte, -128 to 127


class DBCaseConfig(ABC):
    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError


class EmptyDBCaseConfig(BaseModel, DBCaseConfig):
    null: str | None = None
    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}


class WeaviateIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    ef: int | None = -1
    efConstruction: int | None = None
    maxConnections: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2-squared"
        elif self.metric_type == MetricType.IP:
            return "dot"
        return "cosine"

    def index_param(self) -> dict:
        if self.maxConnections is not None and self.efConstruction is not None:
            params = {"distance": self.parse_metric(), "maxConnections": self.maxConnections, "efConstruction": self.efConstruction}
        else:
            params = {"distance": self.parse_metric()}
        return params

    def search_param(self) -> dict:
        return {
            "ef": self.ef,
        }


class MilvusIndexConfig(BaseModel):
    """Base config for milvus"""

    index: IndexType
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        if self.metric_type == MetricType.COSINE:
            return MetricType.L2.value
        return self.metric_type.value


class HNSWConfig(MilvusIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }


class DISKANNConfig(MilvusIndexConfig, DBCaseConfig):
    search_list: int | None = None
    index: IndexType = IndexType.DISKANN

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"search_list": self.search_list},
        }


class IVFFlatConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe},
        }


class FLATConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {},
        }


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


class ElasticsearchIndexConfig(BaseModel, DBCaseConfig):
    element_type: ESElementType = ESElementType.float
    index: IndexType = IndexType.ES_HNSW  # ES only support 'hnsw'

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        params = {
            "type": "dense_vector",
            "index": True,
            "element_type": self.element_type.value,
            "similarity": self.parse_metric(),
            "index_options": {
                "type": self.index.value,
                "m": self.M,
                "ef_construction": self.efConstruction
            }
        }
        return params

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }
