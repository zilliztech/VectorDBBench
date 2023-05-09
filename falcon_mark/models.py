from typing import Any
from enum import IntEnum
from pydantic import BaseModel
from .metric import Metric

class CaseType(IntEnum):
    LoadLDim: 10000
    LoadSDim: 10001

    PerformanceLZero: 10002
    PerformanceMZero: 10003
    PerformanceSZero: 10004

    PerformanceLLow: 10005
    PerformanceMLow: 10006
    PerformanceSLow: 10007
    PerformanceLHigh: 10008
    PerformanceMHigh: 10009
    PerformanceSHigh: 10010


class CustomizedCase(BaseModel):
    pass
    # TODO


class DB(IntEnum):
    """Database types

    Examples:
        >>> DB.Milvus
        100
        >>> DB.Milvus.name
        "Milvus"
    """
    Milvus: 100
    ZillizCloud: 101

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

    def __repr__(self) -> str:
        return f"MilvusConfig<uri={self.uri}>"


class ZillizCloudConfig(BaseModel):
    uri:        str
    user:       str
    password:   str

    def __repr__(self) -> str:
        return f"ZillizCloudConfig<uri={self.uri}, user={self.user}>"


db2config = {
    "Milvus": MilvusConfig,
    "ZillizCloud": ZillizCloudConfig,
}


class CaseConfig(BaseModel):
    """dataset, test cases, filter rate, params"""
    case_id: CaseType
    custom_case: dict
    params: dict = None

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
