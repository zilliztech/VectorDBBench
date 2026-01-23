from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class AliSQLConfigDict(TypedDict):
    """These keys will be directly used as kwargs in alisql connection string,
    so the names must match exactly alisql API"""

    user: str
    password: str
    host: str
    port: int
    database: str


class AliSQLConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "vectordbbench"

    def to_dict(self) -> AliSQLConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
            "database": self.database,
        }


class AliSQLIndexConfig(BaseModel):
    """Base config for AliSQL"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)


class AliSQLHNSWConfig(AliSQLIndexConfig, DBCaseConfig):
    M: int | None
    ef_search: int | None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "M": self.M,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "ef_search": self.ef_search,
        }


_alisql_case_config = {
    IndexType.HNSW: AliSQLHNSWConfig,
}
