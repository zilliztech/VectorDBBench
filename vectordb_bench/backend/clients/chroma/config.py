from chromadb.config import Settings
from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class ChromaConfig(DBConfig):
    user: str | None = None
    password: SecretStr | None
    host: SecretStr = "localhost"
    port: int = 8000

    def to_dict(self) -> dict:
        config = {
            "host": self.host.get_secret_value(),
            "port": self.port,
        }
        if self.password and self.user:
            config["settings"] = Settings(
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                    chroma_client_auth_credentials=f"{self.user}:{self.password}",
                )
            )
        return config


class ChromaIndexConfig(ChromaConfig, DBCaseConfig):
    metric_type: MetricType = "cosine"
    m: int = 16
    ef_construct: int = 100
    ef_search: int | None = 100

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.IP:
            return "ip"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        raise ValueError("Unsupported metric type: %s" % self.metric_type)

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
        return {"ef_search": self.ef_search}
