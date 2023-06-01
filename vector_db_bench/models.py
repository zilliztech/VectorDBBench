import logging
import pathlib
from datetime import date
from typing import Type, Self
from enum import Enum

import ujson

from .base import BaseModel
from . import RESULTS_LOCAL_DIR
from .metric import Metric
from .backend.clients import (
    VectorDB,
    Milvus,
    Weaviate,
    ZillizCloud,
    Elasticsearch,
    Qdrant,
    Pinecone,
)

from .backend.clients.db_config import (
    DBConfig,
    MilvusConfig,
    ZillizCloudConfig,
    WeaviateConfig,
    ElasticsearchConfig,
    QdrantConfig,
    PineconeConfig,
)

from .backend.clients.db_case_config import (
    DBCaseConfig, # base class
    IndexType, MetricType, # Const
    HNSWConfig, DISKANNConfig, IVFFlatConfig, FLATConfig, # Milvus Configs
    AutoIndexConfig, # ZillizCound configs
    WeaviateIndexConfig, # Weaviate configs
    ElasticsearchIndexConfig, # Elasticsearch configs
    EmptyDBCaseConfig,
)


log = logging.getLogger(__name__)


class CaseType(Enum):
    """
    Value will be displayed in UI
    """

    LoadLDim = "Capacity Test(Large-dim)"
    LoadSDim = "Capacity Test(Small-dim)"

    PerformanceLZero = "Search Performance Test(Large Dataset)"
    PerformanceMZero = "Search Performance Test(Medium Dataset)"
    PerformanceSZero = "Search Performance Test(Small Dataset)"

    PerformanceLLow = (
        "Filtering Search Performance Test (Large Dataset, Low Filtering Rate)"
    )
    PerformanceMLow = (
        "Filtering Search Performance Test (Medium Dataset, Low Filtering Rate)"
    )
    PerformanceSLow = (
        "Filtering Search Performance Test (Small Dataset, Low Filtering Rate)"
    )
    PerformanceLHigh = (
        "Filtering Search Performance Test (Large Dataset, High Filtering Rate)"
    )
    PerformanceMHigh = (
        "Filtering Search Performance Test (Medium Dataset, High Filtering Rate)"
    )
    PerformanceSHigh = (
        "Filtering Search Performance Test (Small Dataset, High Filtering Rate)"
    )


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
    numCandidates = "num_candidates"


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
    Elasticsearch = "Elasticsearch"
    Qdrant = "Qdrant"
    Pinecone = "Pinecone"


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
        if self == DB.Elasticsearch:
            return ElasticsearchIndexConfig
        return EmptyDBCaseConfig


_db2config = {
    DB.Milvus: MilvusConfig,
    DB.ZillizCloud: ZillizCloudConfig,
    DB.Weaviate: WeaviateConfig,
    DB.Elasticsearch: ElasticsearchConfig,
    DB.Qdrant: QdrantConfig,
    DB.Pinecone: PineconeConfig,
}

_db2client = {
    DB.Milvus: Milvus,
    DB.ZillizCloud: ZillizCloud,
    DB.Weaviate: Weaviate,
    DB.Elasticsearch: Elasticsearch,
    DB.Qdrant: Qdrant,
    DB.Pinecone: Pinecone,
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
    case_id: CaseType
    custom_case: dict | None = None


class TaskConfig(BaseModel):
    db: DB
    db_config: DBConfig
    db_case_config: DBCaseConfig
    case_config: CaseConfig

    @property
    def db_name(self):
        db = self.db.value
        db_label = self.db_config.db_label
        return f"{db}-{db_label}" if db_label else db


class CaseResult(BaseModel):
    metrics: Metric
    task_config: TaskConfig


class TestResult(BaseModel):
    """ ROOT/result_{date.today()}_{task_label}.json """
    run_id: str
    task_label: str
    results: list[CaseResult]

    def write_file(self):
        result_dir = RESULTS_LOCAL_DIR
        if not result_dir.exists():
            log.info(f"local result directory not exist, creating it: {result_dir}")
            result_dir.mkdir(parents=True)

        file_name = f'result_{date.today().strftime("%Y%m%d")}_{self.task_label}.json'
        result_file = result_dir.joinpath(file_name)
        if result_file.exists():
            log.warning(f"Replacing existing result with the same file_name: {result_file}")

        log.info(f"write results to disk {result_file}")
        with open(result_file, 'w') as f:
            b = self.json(exclude={'db_config': {'password', 'api_key'}})
            f.write(b)


    @classmethod
    def read_file(cls, full_path: pathlib.Path) -> Self:
        if not full_path.exists():
            raise ValueError(f"No such file: {full_path}")

        with open(full_path) as f:
            test_result = ujson.loads(f.read())
            if "task_label" not in test_result:
                test_result['task_label'] = test_result['run_id']

            for case_result in test_result["results"]:
                task_config = case_result.get("task_config")
                db = DB(task_config.get("db"))
                task_config["db_config"] = db.config(**task_config["db_config"])
                task_config["db_case_config"] = db.case_config_cls(
                    index=task_config["db_case_config"].get("index", None)
                )(**task_config["db_case_config"])

                case_result["task_config"] = task_config
            c = TestResult.validate(test_result)
            return c
