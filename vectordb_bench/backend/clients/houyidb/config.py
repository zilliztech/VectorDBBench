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
    # Reject a run when the server does not report, or reports values other
    # than, the requested HNSW build parameters. Disable only for servers
    # older than the /v1/index/status build-parameter fields; doing so means
    # the recorded m/ef_construction may differ from the effective graph.
    verify_build_params: bool = True

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "use_tls": self.use_tls,
            "transport": self.transport,
            "api_key": self.api_key.get_secret_value() if self.api_key else None,
            "timeout_sec": self.timeout_sec,
            "http_port": self.http_port,
            "verify_build_params": self.verify_build_params,
        }


class HouyiDBIndexConfig(BaseModel, DBCaseConfig):
    """HNSW build and search settings.

    HouyiDB's main index is built with cosine distance; the client rejects
    datasets that ask for any other metric instead of benchmarking a
    different distance than the dataset ground truth uses.
    """

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
