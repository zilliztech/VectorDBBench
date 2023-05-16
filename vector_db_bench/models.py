from typing import Any, Type
from enum import IntEnum, Enum
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

from .metric import Metric
from .db_config import (
    DBConfig,
    MilvusConfig,
    ZillizCloudConfig,
    WeaviateConfig
)

from .db_case_config import (
    DBCaseConfig, # base class
    IndexType, MetricType, # Const
    HNSWConfig, DISKANNConfig, IVFFlatConfig, FLATConfig, # Milvus Configs
    AutoIndexConfig, # ZillizCound configs
    WeaviateIndexConfig, # Weaviate configs
    EmptyDBCaseConfig,
)


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


class CaseConfigParamType(Enum):
    """
    Value will be the key of CaseConfig.params and displayed in UI
    """

    IndexType = "IndexType"
    M = "M"
    EFConstruction = "efConstruction"
    EF = "ef"
    SearchList = "search_list"
    Nlist = "nlist"
    Nprobe = "nprobe"


class DB(Enum):
    """Database types

    Examples:
        >>> DB.Milvus
        <DB.Milvus: 'Milvus'>
        >>> DB.Milvus.value
        "Milvus"
        >>> DB.Milvus.name
        "Milvus"
    """

    Milvus = "Milvus"
    ZillizCloud = "ZillizCloud"
    Weaviate = "Weaviate"

    @property
    def config(self) -> Type[DBConfig]:
        """Get configs of the DB
        Examples:
            >>> config_cls = DB.Milvus.config
            >>> config_cls(uri="localhost:19530")

        Returns:
            None, if the database not in the db2config
        """
        return _db2config.get(self)

    def case_config_cls(self, index: IndexType | None = None) -> Type[DBCaseConfig]:
        """Get case config class of the DB
        Examples:
            >>> case_config_cls = DB.Milvus.case_config_cls(IndexType.HNSW)
            >>> hnsw_config = {
            ...     M: 8,
            ...     efConstruction: 12,
            ...     ef: 8,
            >>> }
            >>> milvus_hnsw_config = case_config_cls(**hnsw_config)
        """
        if self == DB.Milvus:
            assert index is not None, "Please provide valid index for DB Milvus"
            return _milvus_case_config.get(index)
        if self == DB.ZillizCloud:
            return AutoIndexConfig
        if self == DB.Weaviate:
            return WeaviateIndexConfig
        return EmptyDBCaseConfig


_db2config = {
    DB.Milvus: MilvusConfig,
    DB.ZillizCloud: ZillizCloudConfig,
    DB.Weaviate: WeaviateConfig,
}


_milvus_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.DISKANN: DISKANNConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.Flat: FLATConfig,
}


class CustomizedCase(BaseModel):
    pass


class CaseConfig(BaseModel):
    """cases, dataset, test cases, filter rate, params"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: CaseType
    custom_case: dict | None = None


class TaskConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: DB
    db_config: DBConfig
    db_case_config: DBCaseConfig
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
