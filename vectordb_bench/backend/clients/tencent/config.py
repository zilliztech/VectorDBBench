from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType, IndexType
from tcvectordb.model.enum import IndexType as TIndexType, MetricType as TMetricType


class TencentVDBConfig(DBConfig, BaseModel):
    url: SecretStr = ""
    username: SecretStr = "root"
    key: SecretStr = ""
    m: str = 30
    efconstruction: str = 200
    ef: str = 100
    replicas: str = 2  # must more than 2
    shard: str = 1

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "username": self.username.get_secret_value(),
            "key": self.key.get_secret_value(),
            "m": int(self.m),
            "efconstruction": int(self.efconstruction),
            "ef": int(self.ef),
            "replicas": int(self.replicas),
            "shard": int(self.shard),
        }


class TencentVDBIndexConfig(DBCaseConfig, BaseModel):
    # HNSW, FLAT, IVF_FLAT, IVF_PQ, IVF_SQ4, IVF_SQ8, IVF_SQ16
    indexType: TIndexType = TIndexType.HNSW
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return TMetricType.COSINE

        if self.metric_type == MetricType.IP:
            return TMetricType.IP

        if self.metric_type == MetricType.L2:
            return TMetricType.L2

        return TMetricType.L2

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }

    def search_param(self) -> dict:
        return {}
