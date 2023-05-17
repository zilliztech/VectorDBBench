import logging
import pathlib
from datetime import date
from typing import Type, Self
from enum import Enum

import ujson
from pydantic import BaseModel, ConfigDict

from . import RESULTS_LOCAL_DIR
from .metric import Metric
from .backend.clients import (
    VectorDB,
    Milvus,
    Weaviate,
    ZillizCloud,
)

from .backend.clients.db_config import (
    DBConfig,
    MilvusConfig,
    ZillizCloudConfig,
    WeaviateConfig
)

from .backend.clients.db_case_config import (
    DBCaseConfig, # base class
    IndexType, MetricType, # Const
    HNSWConfig, DISKANNConfig, IVFFlatConfig, FLATConfig, # Milvus Configs
    AutoIndexConfig, # ZillizCound configs
    WeaviateIndexConfig, # Weaviate configs
    EmptyDBCaseConfig,
)


log = logging.getLogger(__name__)


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
    MaxConnections = "maxConnections"


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

    @property
    def init_cls(self) -> Type[VectorDB]:
        return _db2client.get(self)

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

_db2client = {
    DB.Milvus: Milvus,
    DB.ZillizCloud: ZillizCloud,
    DB.Weaviate: Weaviate,
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: Metric
    result_id: int
    task_config: TaskConfig


class TestResult(BaseModel):
    """ ROOT/result_{date.today()}_{run_id}.json """
    run_id: int
    results: list[CaseResult]

    def write_file(self):
        result_dir = pathlib.Path(RESULTS_LOCAL_DIR)
        if not result_dir.exists():
            result_dir.mkdir(parents=True)

        result_file = result_dir.joinpath(f'result_{date.today().strftime("%Y%m%d")}_{self.run_id}.json')
        if result_file.exists():
            raise ValueError(f"try to write to existing file: {result_file}")

        with open(result_file, 'w') as f:
            b = self.model_dump_json()
            f.write(b)


    @classmethod
    def read_file(cls, name: str) -> Self:
        fname = pathlib.Path(RESULTS_LOCAL_DIR, name)
        if not fname.exists():
            raise ValueError(f"No such file: {fname}")

        with open(fname) as f:
            test_result = ujson.loads(f.read())

            for case_result in test_result['results']:
                task_config = case_result.get('task_config')
                db = DB(task_config.get('db'))
                task_config['db_config'] = db.config(**task_config['db_config'])
                task_config['db_case_config'] = db.case_config_cls(index=task_config['db_case_config']['index'])(**task_config['db_case_config'])

                case_result['task_config'] = task_config
            c = TestResult.model_validate(test_result)
            return c
