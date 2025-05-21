from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class MariaDBConfigDict(TypedDict):
    """These keys will be directly used as kwargs in mariadb connection string,
    so the names must match exactly mariadb API"""

    user: str
    password: str
    host: str
    port: int


class MariaDBConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 3306

    def to_dict(self) -> MariaDBConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
        }


class MariaDBIndexConfig(BaseModel):
    """Base config for MariaDB"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)


class MariaDBHNSWConfig(MariaDBIndexConfig, DBCaseConfig):
    M: int | None
    ef_search: int | None
    index: IndexType = IndexType.HNSW
    storage_engine: str = "InnoDB"
    max_cache_size: int | None

    def index_param(self) -> dict:
        return {
            "storage_engine": self.storage_engine,
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "M": self.M,
            "max_cache_size": self.max_cache_size,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "ef_search": self.ef_search,
        }


_mariadb_case_config = {
    IndexType.HNSW: MariaDBHNSWConfig,
}
