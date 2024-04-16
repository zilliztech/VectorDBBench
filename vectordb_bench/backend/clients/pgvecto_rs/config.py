from typing import Literal
from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgVectoRSConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str
        }

class PgVectoRSIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif self.metric_type == MetricType.IP:
            return "vector_dot_ops"
        return "vector_cos_ops"

    def parse_metric_fun_op(self) -> str:
        if self.metric_type == MetricType.L2:
            return "<->"
        elif self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

class PgVectoRSQuantConfig(PgVectoRSIndexConfig):
    quantizationType: Literal["trivial", "scalar", "product"]
    quantizationRatio: None | Literal["x4", "x8", "x16", "x32", "x64"]

    def parse_quantization(self) -> str:
        if self.quantizationType == "trivial":
            return "quantization = { trivial = { } }"
        elif self.quantizationType == "scalar":
            return "quantization = { scalar = { } }"
        else:
            return f'quantization = {{ product = {{ ratio = "{self.quantizationRatio}" }} }}'


class HNSWConfig(PgVectoRSQuantConfig):
    M: int
    efConstruction: int
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        options = f"""
[indexing.hnsw]
m = {self.M}
ef_construction = {self.efConstruction}
{self.parse_quantization()}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}


class IVFFlatConfig(PgVectoRSQuantConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        options = f"""
[indexing.ivf]
nlist = {self.nlist}
nsample = {self.nprobe if self.nprobe else 10}
{self.parse_quantization()}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}

class IVFFlatSQ8Config(PgVectoRSIndexConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFSQ8

    def index_param(self) -> dict:
        options = f"""
[indexing.ivf]
nlist = {self.nlist}
nsample = {self.nprobe if self.nprobe else 10}
quantization = {{ scalar = {{ }} }}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}

class FLATConfig(PgVectoRSQuantConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        options = f"""
[indexing.flat]
{self.parse_quantization()}
"""
        return {"options": options, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"metrics_op": self.parse_metric_fun_op()}


_pgvecto_rs_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.IVFSQ8: IVFFlatSQ8Config,
    IndexType.Flat: FLATConfig,
}
