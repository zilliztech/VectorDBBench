from abc import abstractmethod
from typing import LiteralString, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgVectorScaleConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class PgVectorScaleConfig(DBConfig):
    user_name: SecretStr = SecretStr("postgres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> PgVectorScaleConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class PgVectorScaleIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "vector_cosine_ops"
        return ""

    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.COSINE:
            return "<=>"
        return ""

    def parse_metric_fun_str(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine_distance"
        return ""

    @abstractmethod
    def index_param(self) -> dict: ...

    @abstractmethod
    def search_param(self) -> dict: ...

    @abstractmethod
    def session_param(self) -> dict: ...


class PgVectorScaleStreamingDiskANNConfig(PgVectorScaleIndexConfig):
    index: IndexType = IndexType.STREAMING_DISKANN
    storage_layout: str | None
    num_neighbors: int | None
    search_list_size: int | None
    max_alpha: float | None
    num_dimensions: int | None
    num_bits_per_dimension: int | None
    query_search_list_size: int | None
    query_rescore: int | None

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "options": {
                "storage_layout": self.storage_layout,
                "num_neighbors": self.num_neighbors,
                "search_list_size": self.search_list_size,
                "max_alpha": self.max_alpha,
                "num_dimensions": self.num_dimensions,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> dict:
        return {
            "diskann.query_search_list_size": self.query_search_list_size,
            "diskann.query_rescore": self.query_rescore,
        }


_pgvectorscale_case_config = {
    IndexType.STREAMING_DISKANN: PgVectorScaleStreamingDiskANNConfig,
}
