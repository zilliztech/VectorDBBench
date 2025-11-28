from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class TurbopufferConfig(DBConfig):
    api_key: SecretStr
    region: str = "us-east-1"
    namespace: str = "vdbbench"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value() if self.api_key else "",
            "region": self.region,
            "namespace": self.namespace,
        }


class TurbopufferIndexConfig(DBCaseConfig, BaseModel):
    """Base config for turbopuffer"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine_distance"
        if self.metric_type == MetricType.L2:
            return "euclidean_squared"
        if self.metric_type == MetricType.IP:
            return "dot_product"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
