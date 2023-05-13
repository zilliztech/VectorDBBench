from typing import Any
from enum import IntEnum, Enum
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
from .metric import Metric


class CaseType(Enum):
    """
    Value will be displayed in UI
    """
    LoadLDim = "Capacity-1"
    LoadSDim = "Capacity-2"

    PerformanceLZero = "ANNS-1"
    PerformanceMZero = "ANNS-2"
    PerformanceSZero = "ANNS-3"

    PerformanceLLow = "Filter-1"
    PerformanceMLow = "Filter-2"
    PerformanceSLow = "Filter-3"
    PerformanceLHigh = "Filter-4"
    PerformanceMHigh = "Filter-5"
    PerformanceSHigh = "Filter-6"


class IndexType(str, Enum):
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    Flat = "FLAT"


class MetricType(str, Enum):
    L2 = "L2"
    COSIN = "COSIN"
    IP = "IP"


class CustomizedCase(BaseModel):
    pass
    # TODO


class CaseConfigParamType(Enum):
    """
    Name will be displayed in UI
    Value will be the key of CaseConfig.params
    """
    IndexType = "IndexType"
    M = "M"
    EFConstruction = "efConstruction"
    EF = "ef"
    SearchList = "search_list"
    Nlist = "nlist"
    Nprobe = "nprobe"


class DBConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

class DBCaseConfig(ABC):
    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError


class DB(IntEnum):
    """Database types

    Examples:
        >>> DB.Milvus
        100
        >>> DB.Milvus.name
        "Milvus"
    """

    Milvus = 100
    ZillizCloud = 101

    @property
    def config(self) -> DBConfig:
        """Get configs of the DB type
        Examples:
            >>> DB.Milvus.config

        Returns:
            None, if the database not in the db2config
        """
        return db2config.get(self.name, None)

class MilvusConfig(DBConfig, BaseModel):
    uri: str = "http://localhost:19530"

    def to_dict(self) -> dict:
        return {"uri": self.uri}

class ZillizCloudConfig(DBConfig, BaseModel):
    uri: str
    user: str
    password: str

    def to_dict(self) -> dict:
        return {"uri": self.uri, "user": self.user, "password": self.password}


db2config = {
    "Milvus": MilvusConfig,
    "ZillizCloud": ZillizCloudConfig,
}


class CaseConfig(BaseModel):
    """cases, dataset, test cases, filter rate, params"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: CaseType
    db_case_config: DBCaseConfig

    custom_case: dict | None = None


class TaskConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: DB
    db_config: DBConfig
    case_config: CaseConfig


class CaseResult(BaseModel):
    result_id: int
    case_config: CaseConfig
    output_path: str

    metrics: list[Metric]

    def append_to_disk(self):
        pass


class TestResult(BaseModel):
    run_id: int
    results: list[CaseResult]
