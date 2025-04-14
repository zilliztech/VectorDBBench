from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


# Allowing `api_key` to be left empty, to ensure compatibility with the open-source Qdrant.
class QdrantConfig(DBConfig):
    url: SecretStr
    api_key: SecretStr

    def to_dict(self) -> dict:
        api_key = self.api_key.get_secret_value()
        if len(api_key) > 0:
            return {
                "url": self.url.get_secret_value(),
                "api_key": self.api_key.get_secret_value(),
                "prefer_grpc": True,
            }
        return {
            "url": self.url.get_secret_value(),
        }


class QdrantIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "Euclid"

        if self.metric_type == MetricType.IP:
            return "Dot"

        return "Cosine"

    def index_param(self) -> dict:
        return {"distance": self.parse_metric()}

    def search_param(self) -> dict:
        return {}
