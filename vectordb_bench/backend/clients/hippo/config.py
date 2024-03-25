from pydantic import BaseModel, Field, SecretStr
from transwarp_hippo_api.hippo_type import IndexType
from transwarp_hippo_api.hippo_type import MetricType as HippoMetricType

from ..api import DBCaseConfig, DBConfig, MetricType


class HippoConfig(DBConfig):
    ip: SecretStr = ""
    port: SecretStr = "18902"
    username: SecretStr = "shiva"
    password: SecretStr = "shiva"
    number_of_shards: int = Field(default=1, ge=1)
    number_of_replicas: int = Field(default=1, ge=1)
    insert_batch_size: int = Field(default=100, ge=1)

    def to_dict(self) -> dict:
        return {
            "host_port": [
                f"{self.ip.get_secret_value()}:{self.port.get_secret_value()}"
            ],
            "username": self.username.get_secret_value(),
            "pwd": self.password.get_secret_value(),
            "number_of_shards": self.number_of_shards,
            "number_of_replicas": self.number_of_replicas,
            "insert_batch_size": self.insert_batch_size,
        }


class HippoIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW  # HNSW, FLAT, IVF_FLAT, IVF_SQ, IVF_PQ, ANNOY
    metric_type: MetricType | None = None
    M: int = 30  # [4,96]
    ef_construction: int = 360  # [8, 512]
    ef_search: int = 100  # [topk, 32768]
    nlist: int = 1024  # [1,65536]
    nprobe: int = 64  # [1, nlist]
    m: int = 16  # divisible by dim
    nbits: int = 8  # [1, 16]
    k_factor: int = 100  # [10, 1000]

    def parse_metric(self) -> HippoMetricType:
        if self.metric_type == MetricType.COSINE:
            return HippoMetricType.COSINE
        if self.metric_type == MetricType.IP:
            return HippoMetricType.IP
        if self.metric_type == MetricType.L2:
            return HippoMetricType.L2
        return ""

    def index_param(self) -> dict:
        return {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "m": self.m,
            "nbits": self.nbits,
        }

    def search_param(self) -> dict:
        return {
            "ef_search": self.ef_search,
            "nprobe": self.nprobe,
            "k_factor": self.k_factor,
        }
