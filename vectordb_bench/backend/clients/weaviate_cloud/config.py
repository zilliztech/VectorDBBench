from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class WeaviateConfig(DBConfig):
    url: SecretStr
    api_key: SecretStr
    no_auth: bool | None = False
    # optional gRPC endpoint like "localhost:50051"
    grpc_url: SecretStr | None = None

    # Backward-compat method used by older code paths; keep for compatibility
    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "auth_client_secret": self.api_key.get_secret_value(),
            "no_auth": self.no_auth,
            "grpc_url": self.grpc_url.get_secret_value() if self.grpc_url else None,
        }

    # Helpers for v4 client wiring
    def host_port(self) -> tuple[str, int]:
        from urllib.parse import urlparse

        url = self.url.get_secret_value()
        u = urlparse(url)
        host = u.hostname or "localhost"
        if u.port:
            port = u.port
        else:
            port = 443 if (u.scheme or "http").lower() == "https" else 80
        return host, port

    def grpc_host_port(self) -> tuple[str, int] | None:
        if not self.grpc_url:
            return None
        value = self.grpc_url.get_secret_value()
        if not value:
            return None
        if ":" in value:
            host, port_str = value.split(":", 1)
            try:
                return host, int(port_str)
            except ValueError:
                return host, 50051
        return value, 50051


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
