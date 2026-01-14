from enum import Enum

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class AstraDBConfig(DBConfig, BaseModel):
    api_endpoint: str = "https://<database-id>-<region>.apps.astra.datastax.com"
    token: SecretStr = "<your-astra-token>"
    namespace: str = "default_keyspace"

    def to_dict(self) -> dict:
        return {
            "api_endpoint": self.api_endpoint,
            "token": self.token.get_secret_value(),
            "namespace": self.namespace,
        }


class AstraDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW  # AstraDB uses vector search
    metric_type: MetricType = MetricType.COSINE

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"  # Default to cosine similarity

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
        }

    def search_param(self) -> dict:
        return {}
