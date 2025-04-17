from abc import abstractmethod
from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class ClickhouseConfigDict(TypedDict):
    user: str
    password: str
    host: str
    port: int
    database: str
    secure: bool


class ClickhouseConfig(DBConfig):
    user_name: str = "clickhouse"
    password: SecretStr
    host: str = "localhost"
    port: int = 8123
    db_name: str = "default"
    secure: bool = False

    def to_dict(self) -> ClickhouseConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "database": self.db_name,
            "user": self.user_name,
            "password": pwd_str,
            "secure": self.secure,
        }


class ClickhouseIndexConfig(BaseModel, DBCaseConfig):

    metric_type: MetricType | None = None
    vector_data_type: str | None = "Float32"  # Data type of vectors. Can be Float32 or Float64 or BFloat16
    create_index_before_load: bool = True
    create_index_after_load: bool = False

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value

    def parse_metric_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "L2Distance"
        if self.metric_type == MetricType.COSINE:
            return "cosineDistance"
        return "cosineDistance"

    @abstractmethod
    def session_param(self):
        pass


class ClickhouseHNSWConfig(ClickhouseIndexConfig):
    M: int | None  # Default in clickhouse in 32
    efConstruction: int | None  # Default in clickhouse in 128
    ef: int | None = None
    index: IndexType = IndexType.HNSW
    quantization: str | None = "bf16"  # Default is bf16. Possible values are f64, f32, f16, bf16, or i8
    granularity: int | None = 10_000_000  # Size of the index granules. By default, in CH it's equal 10.000.000

    def index_param(self) -> dict:
        return {
            "vector_data_type": self.vector_data_type,
            "metric_type": self.parse_metric_str(),
            "index_type": self.index.value,
            "quantization": self.quantization,
            "granularity": self.granularity,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric_str(),
            "params": {"ef": self.ef},
        }

    def session_param(self) -> dict:
        return {
            "allow_experimental_vector_similarity_index": 1,
        }
