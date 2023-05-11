from typing import Any
from enum import IntEnum, Enum
from pydantic import BaseModel
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

    def config(self) -> Any:
        """Get configs of the DB type
        Examples:
            >>> DB.Milvus.config()

        Returns:
            None, if the database not in the db2config
        """
        return db2config.get(self.name, None)


class MilvusConfig(BaseModel):
    uri: str


class ZillizCloudConfig(BaseModel):
    uri: str
    user: str
    password: str


db2config = {
    "Milvus": MilvusConfig,
    "ZillizCloud": ZillizCloudConfig,
}

class DBCaseConfig(ABC):
    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError


    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError


class CaseConfig(BaseModel):
    """dataset, test cases, filter rate, params"""

    case_id: CaseType
    custom_case: dict
    db_case_config: DBCaseConfig


class TaskConfig(BaseModel):
    db: DB
    db_config: Any
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
