from pydantic import SecretStr

from ..api import DBConfig, DBCaseConfig, MetricType


class ChromaConfig(DBConfig):
    host: SecretStr = "localhost"
    port: int = 8000

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port
        }


class ChromaIndexConfig(ChromaConfig, DBCaseConfig):
    metric_type: MetricType = "cosine"
    m: int = 16
    ef_construct: int = 100
    ef_search: int | None = 100

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        elif self.metric_type == MetricType.IP:
            return "ip"
        elif self.metric_type == MetricType.COSINE:
            return "cosine"
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def index_param(self):
        return {
            "hnsw": {
                "space": self.parse_metric(),
                "max_neighbors": self.m,
                "ef_construction": self.ef_construct,
                "ef_search": self.search_param().get("ef_search", 100),
            }
        }

    def search_param(self) -> dict:
        return {
            "ef_search": self.ef_search
        }
