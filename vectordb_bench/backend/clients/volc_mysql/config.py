#!/usr/bin/env python3
from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class VolcMySQLConfigDict(TypedDict):
    """Keys used directly as kwargs in mysql.connector.connect;
    names must match mysql-connector-python's API."""

    user: str
    password: str
    host: str
    port: int


class VolcMySQLConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 3306

    def to_dict(self) -> VolcMySQLConfigDict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": self.password.get_secret_value(),
        }


class VolcMySQLIndexConfig(BaseModel):
    """Base index config for VolcMySQL"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)


class VolcMySQLHNSWConfig(VolcMySQLIndexConfig, DBCaseConfig):
    M: int | None = None
    ef_search: int | None = None
    ef_construction: int | None = None
    quant_algorithm: str | None = None
    quant_type: str | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "quant_algorithm": self.quant_algorithm,
            "quant_type": self.quant_type,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "ef_search": self.ef_search,
        }


_volcmysql_case_config = {
    IndexType.HNSW: VolcMySQLHNSWConfig,
}
