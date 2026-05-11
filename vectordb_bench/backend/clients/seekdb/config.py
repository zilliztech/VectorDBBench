from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class SeekDBConfigDict(TypedDict):
    user: str
    host: str
    port: int
    password: str
    database: str


class SeekDBConfig(DBConfig):
    user: SecretStr = SecretStr("root")
    password: SecretStr
    host: str
    port: int = 3306
    database: str

    def to_dict(self) -> SeekDBConfigDict:
        return {
            "user": self.user.get_secret_value(),
            "host": self.host,
            "port": self.port,
            "password": self.password.get_secret_value(),
            "database": self.database,
        }


class SeekDBIndexConfig(BaseModel):
    index: IndexType
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.IP:
            return "inner_product"
        return "cosine"

    def parse_metric_func_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        if self.metric_type == MetricType.IP:
            return "negative_inner_product"
        return "cosine_distance"


class SeekDBHNSWConfig(SeekDBIndexConfig, DBCaseConfig):
    m: int
    ef_construction: int
    ef_search: int
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "m": self.m,
                "ef_construction": self.ef_construction,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric_func_str(),
            "params": {"ef_search": self.ef_search},
        }


_seekdb_case_config = {
    IndexType.HNSW: SeekDBHNSWConfig,
}
