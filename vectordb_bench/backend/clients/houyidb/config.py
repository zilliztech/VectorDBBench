"""HouyiDB connection and index settings for VectorDBBench."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class HouyiDBConfig(DBConfig):
    """Connection settings for a HouyiDB node."""

    host: str
    port: int = 15432
    http_port: int | None = 17000
    use_tls: bool = False
    transport: str = "pgwire"
    api_key: SecretStr | None = None
    timeout_sec: int = 600

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "use_tls": self.use_tls,
            "transport": self.transport,
            "api_key": self.api_key.get_secret_value() if self.api_key else None,
            "timeout_sec": self.timeout_sec,
            "http_port": self.http_port,
        }


class HouyiDBIndexConfig(BaseModel, DBCaseConfig):
    """HNSW build and search settings."""

    metric_type: MetricType | None = MetricType.COSINE
    m: int = 24
    ef_construction: int = 400
    ef_search: int = 256

    def index_param(self) -> dict:
        return {
            "metric_type": self.metric_type.name if self.metric_type else MetricType.COSINE.name,
            "m": self.m,
            "ef_construction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {"ef_search": self.ef_search}
