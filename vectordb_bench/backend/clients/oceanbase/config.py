from typing import TypedDict

from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class OceanBaseConfigDict(TypedDict):
    user: str
    host: str
    port: str
    password: str
    database: str


class OceanBaseConfig(DBConfig):
    user: SecretStr = SecretStr("root@perf")
    password: SecretStr
    host: str
    port: int
    database: str

    def to_dict(self) -> OceanBaseConfigDict:
        user_str = self.user.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "user": user_str,
            "host": self.host,
            "port": self.port,
            "password": pwd_str,
            "database": self.database,
        }

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if field.name in ["password", "host", "db_label"]:
            return v
        if isinstance(v, str | SecretStr) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class OceanBaseIndexConfig(BaseModel):
    index: IndexType
    metric_type: MetricType | None = None
    lib: str = "vsag"

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2 or (
            self.index == IndexType.HNSW_BQ and self.metric_type == MetricType.COSINE
        ):
            return "l2"
        if self.metric_type == MetricType.IP:
            return "inner_product"
        return "cosine"

    def parse_metric_func_str(self) -> str:
        if self.metric_type == MetricType.L2 or (
            self.index == IndexType.HNSW_BQ and self.metric_type == MetricType.COSINE
        ):
            return "l2_distance"
        if self.metric_type == MetricType.IP:
            return "negative_inner_product"
        return "cosine_distance"


class OceanBaseHNSWConfig(OceanBaseIndexConfig, DBCaseConfig):
    m: int
    efConstruction: int
    ef_search: int | None = None
    index: IndexType

    def index_param(self) -> dict:
        return {
            "lib": self.lib,
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"m": self.m, "ef_construction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {"metric_type": self.parse_metric_func_str(), "params": {"ef_search": self.ef_search}}


class OceanBaseIVFConfig(OceanBaseIndexConfig, DBCaseConfig):
    m: int
    sample_per_nlist: int
    nbits: int | None = None
    nlist: int
    index: IndexType
    ivf_nprobes: int | None = None

    def index_param(self) -> dict:
        if self.index == IndexType.IVFPQ:
            return {
                "lib": "OB",
                "metric_type": self.parse_metric(),
                "index_type": self.index.value,
                "params": {
                    "m": self.m,
                    "sample_per_nlist": self.sample_per_nlist,
                    "nbits": self.nbits,
                    "nlist": self.nlist,
                },
            }
        return {
            "lib": "OB",
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "sample_per_nlist": self.sample_per_nlist,
                "nlist": self.nlist,
            },
        }

    def search_param(self) -> dict:
        return {"metric_type": self.metric_type, "params": {"ivf_nprobes": self.ivf_nprobes}}


_oceanbase_case_config = {
    IndexType.HNSW_SQ: OceanBaseHNSWConfig,
    IndexType.HNSW: OceanBaseHNSWConfig,
    IndexType.HNSW_BQ: OceanBaseHNSWConfig,
    IndexType.IVFFlat: OceanBaseIVFConfig,
    IndexType.IVFPQ: OceanBaseIVFConfig,
    IndexType.IVFSQ8: OceanBaseIVFConfig,
}
