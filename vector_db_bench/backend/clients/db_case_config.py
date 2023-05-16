from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel


class IndexType(str, Enum):
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"


class MetricType(str, Enum):
    L2 = "L2"
    COSIN = "COSIN"
    IP = "IP"


class DBCaseConfig(ABC):
    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError


class EmptyDBCaseConfig(DBCaseConfig):
    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}


class WeaviateIndexConfig(BaseModel, DBCaseConfig):
    metric_type: str | None = None
    ef: int | None = -1
    efConstruction: int | None = None
    maxConnections: int | None = None

    def index_param(self) -> dict:
        if self.maxConnections is not None and self.efConstruction is not None:
            params = {"distance": self.metric_type, "maxConnections": self.maxConnections, "efConstruction": self.efConstruction}
        else:
            params = {"distance": self.metric_type}
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

        if self.metric_type == MetricType.COSIN:
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
            'metric_type': self.parse_metric(),
            'index_type': self.index.value,
            'params': {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }
