"""DXMT connection and index settings for VectorDBBench."""

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

SEARCH_MODES = ("fast", "normal", "accurate")


class DXMTConfig(DBConfig):
    """Where the DXMT server listens. `api_key` is needed only when the server sets DXMT_API_KEY."""

    url: str = "http://localhost:9999"
    api_key: SecretStr | None = None
    timeout_sec: int = 600

    def to_dict(self) -> dict:
        return {
            "url": self.url.rstrip("/"),
            "api_key": self.api_key.get_secret_value() if self.api_key else None,
            "timeout_sec": self.timeout_sec,
        }


class DXMTIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    # Per-query effort: "fast", "normal" or "accurate". None leaves the server default ("normal").
    search_mode: str | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "dot"
        return "cosine"

    def search_param(self) -> dict:
        return {"search_mode": self.search_mode} if self.search_mode else {}


class DXMTAutoIndexConfig(DXMTIndexConfig):
    """The server picks the index and its parameters, as it does for a table created without them."""

    def index_param(self) -> dict:
        return {"distance_metric": self.parse_metric()}


class DXMTHNSWConfig(DXMTIndexConfig):
    """An HNSW index with explicit graph parameters, set per table at creation."""

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 128

    def index_param(self) -> dict:
        return {
            "distance_metric": self.parse_metric(),
            "index_type": "hnsw",
            "hnsw": {"m": self.m, "ef_construction": self.ef_construction, "ef_search": self.ef_search},
        }
