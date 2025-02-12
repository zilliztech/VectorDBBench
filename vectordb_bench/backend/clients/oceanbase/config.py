from typing import Any, Mapping, Optional, Sequence, TypedDict
from pydantic import BaseModel, SecretStr, validator
from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

class OceanBaseConfigDict(TypedDict):
    unix_socket: str
    user: str
    host: str
    port: str
    password: str
    database: str

class OceanBaseConfig(DBConfig):
    user: SecretStr = SecretStr("root@perf")
    password: SecretStr
    unix_socket: str = ""
    host: str
    port: int
    database: str

    def to_dict(self) -> OceanBaseConfigDict:
        user_str = self.user.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "unix_socket": self.unix_socket,
            "user": user_str,
            "host": self.host,
            "port": self.port,
            "password": pwd_str,
            "database": self.database,
        }
    
    @validator("*")
    def not_empty_field(cls, v, field):
        if field.name in ["password", "unix_socket", "host", "db_label"]:
            return v
        if isinstance(v, (str, SecretStr)) and len(v) == 0:
            raise ValueError("Empty string!")
        return v

class OceanBaseIndexConfig(BaseModel):
    metric_type: MetricType | None = None
    lib: str = "vsag"

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        elif self.metric_type == MetricType.IP:
            return "inner_product"
        return "cosine"

    def parse_metric_func_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "negative_inner_product"
        return "cosine_distance"

class OceanBaseHNSWConfig(OceanBaseIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef_search: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "lib": self.lib,
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": { "m": self.M, "ef_construction": self.efConstruction }
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric_func_str(),
            "params": { "ef_search": self.ef_search }
        }
    
_oceanbase_case_config = {
    IndexType.HNSW: OceanBaseHNSWConfig,
}
