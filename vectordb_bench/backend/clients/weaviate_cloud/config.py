from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class WeaviateConfig(DBConfig):
    url: SecretStr
    api_key: SecretStr

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "auth_client_secret": self.api_key.get_secret_value(),
        }


class WeaviateIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    ef: int | None = -1
    efConstruction: int | None = None
    maxConnections: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2-squared"
        if self.metric_type == MetricType.IP:
            return "dot"
        return "cosine"

    def index_param(self) -> dict:
        if self.maxConnections is not None and self.efConstruction is not None:
            params = {
                "distance": self.parse_metric(),
                "maxConnections": self.maxConnections,
                "efConstruction": self.efConstruction,
            }
        else:
            params = {"distance": self.parse_metric()}
        return params

    def search_param(self) -> dict:
        return {
            "ef": self.ef,
        }
