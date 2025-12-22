from abc import abstractmethod
from typing import TypedDict

from pgvecto_rs.types import Flat, Hnsw, IndexOption, Ivf, Quantization
from pgvecto_rs.types.index import QuantizationRatio, QuantizationType
from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgVectorRSConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class PgVectoRSConfig(DBConfig):
    user_name: str = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class PgVectoRSIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    max_parallel_workers: int | None = None
    quantization_type: QuantizationType | None = None
    quantization_ratio: QuantizationRatio | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        if self.metric_type == MetricType.IP:
            return "vector_dot_ops"
        return "vector_cos_ops"

    def parse_metric_fun_op(self) -> str:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    def search_param(self) -> dict:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    @abstractmethod
    def index_param(self) -> dict[str, str]: ...

    @abstractmethod
    def session_param(self) -> dict[str, str | int]: ...


class PgVectoRSHNSWConfig(PgVectoRSIndexConfig):
    index: IndexType = IndexType.HNSW
    m: int | None = None
    ef_search: int | None
    ef_construction: int | None = None

    def index_param(self) -> dict[str, str]:
        if self.quantization_type is None:
            quantization = None
        else:
            quantization = Quantization(typ=self.quantization_type, ratio=self.quantization_ratio)

        option = IndexOption(
            index=Hnsw(
                m=self.m,
                ef_construction=self.ef_construction,
                quantization=quantization,
            ),
            threads=self.max_parallel_workers,
        )
        return {"options": option.dumps(), "metric": self.parse_metric()}

    def session_param(self) -> dict[str, str | int]:
        session_parameters = {}
        if self.ef_search is not None:
            session_parameters["vectors.hnsw_ef_search"] = str(self.ef_search)
        return session_parameters


class PgVectoRSIVFFlatConfig(PgVectoRSIndexConfig):
    index: IndexType = IndexType.IVFFlat
    probes: int | None
    lists: int | None

    def index_param(self) -> dict[str, str]:
        if self.quantization_type is None:
            quantization = None
        else:
            quantization = Quantization(typ=self.quantization_type, ratio=self.quantization_ratio)

        option = IndexOption(
            index=Ivf(nlist=self.lists, quantization=quantization),
            threads=self.max_parallel_workers,
        )
        return {"options": option.dumps(), "metric": self.parse_metric()}

    def session_param(self) -> dict[str, str | int]:
        session_parameters = {}
        if self.probes is not None:
            session_parameters["vectors.ivf_nprobe"] = str(self.probes)
        return session_parameters


class PgVectoRSFLATConfig(PgVectoRSIndexConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict[str, str]:
        if self.quantization_type is None:
            quantization = None
        else:
            quantization = Quantization(typ=self.quantization_type, ratio=self.quantization_ratio)

        option = IndexOption(
            index=Flat(
                quantization=quantization,
            ),
            threads=self.max_parallel_workers,
        )
        return {"options": option.dumps(), "metric": self.parse_metric()}

    def session_param(self) -> dict[str, str | int]:
        return {}


_pgvecto_rs_case_config = {
    IndexType.HNSW: PgVectoRSHNSWConfig,
    IndexType.IVFFlat: PgVectoRSIVFFlatConfig,
    IndexType.Flat: PgVectoRSFLATConfig,
}
