from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType
from transwarp_hippo_api.hippo_type import IndexType, MetricType as HippoMetricType


class HippoConfig(DBConfig):
    ip: SecretStr = ""
    port: SecretStr = ""
    username: SecretStr = ""
    pwd: SecretStr = ""

    def to_dict(self) -> dict:
        return {
            "host_port": [f"{self.ip.get_secret_value()}:{self.port.get_secret_value()}"],
            "username": self.username.get_secret_value(),
            "pwd": self.pwd.get_secret_value(),
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

    def parse_metric(self) -> HippoMetricType:
        # TODO: if support cosine, return cosine
        if self.metric_type == MetricType.COSINE:
            return HippoMetricType.L2
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
        }
