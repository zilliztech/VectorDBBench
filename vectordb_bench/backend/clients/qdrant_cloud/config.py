from pydantic import BaseModel, SecretStr

from ..api import DBConfig, DBCaseConfig, MetricType
from qdrant_client.models import Distance


class QdrantConfig(DBConfig):
    url: SecretStr
    api_key: SecretStr

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "api_key": self.api_key.get_secret_value(),
            "prefer_grpc": True,
        }

class QdrantIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return Distance.EUCLID
        elif self.metric_type == MetricType.IP:
            return Distance.DOT
        return Distance.COSINE

    def index_param(self) -> dict:
        params = {"distance": self.parse_metric()}
        return params

    def search_param(self) -> dict:
        return {}