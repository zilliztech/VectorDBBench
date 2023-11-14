from typing import Literal
from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgVectoRSConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    url: SecretStr
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        url_str = self.url.get_secret_value()
        host, port = url_str.split(":")
        return {
            "host": host,
            "port": port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class PgVectoRSIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    quantizationType: Literal["trivial", "scalar", "product"]
    quantizationRatio: None | Literal["x4", "x8", "x16", "x32", "x64"]

    def parse_quantization(self) -> str:
        if self.quantizationType == "trivial":
            return "quantization = { trivial = { } }"
        elif self.quantizationType == "scalar":
            return "quantization = { scalar = { } }"
        else:
            return f'quantization = {{ product = {{ ratio = "{self.quantizationRatio}" }} }}'

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_ops"
        elif self.metric_type == MetricType.IP:
            return "dot_ops"
        return "cosine_ops"

    def parse_metric_fun_op(self) -> str:
        if self.metric_type == MetricType.L2:
            return "<->"
        elif self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"


class HNSWConfig(PgVectoRSIndexConfig):
    M: int
    efConstruction: int
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        options = f"""
capacity = 1048576
[algorithm.hnsw]
m = {self.M}
ef_construction = {self.efConstruction}
{self.parse_quantization()}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}


class IVFFlatConfig(PgVectoRSIndexConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        options = f"""
capacity = 1048576
[algorithm.ivf]
nlist = {self.nlist}
nprob = {self.nprobe if self.nprobe else 10}
{self.parse_quantization()}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}


class FLATConfig(PgVectoRSIndexConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        options = f"""
capacity = 1048576
[algorithm.flat]
{self.parse_quantization()}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}


_pgvecto_rs_case_config = {
    IndexType.AUTOINDEX: HNSWConfig,
    IndexType.HNSW: HNSWConfig,
    IndexType.DISKANN: HNSWConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.Flat: FLATConfig,
}
