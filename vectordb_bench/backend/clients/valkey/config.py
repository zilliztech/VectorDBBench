from pydantic import BaseModel, Field, SecretStr, model_validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class ValkeyConfig(DBConfig):
    password: SecretStr | None = None
    host: SecretStr
    port: int = Field(default=6379, ge=1, le=65535)
    ssl: bool = True
    insecure_tls: bool = False
    cmd: bool = False
    request_timeout_ms: int = Field(default=600_000, gt=0)
    connection_timeout_ms: int = Field(default=10_000, gt=0)
    collection_name: str = Field(default="vdbbench_valkey", pattern=r"^[A-Za-z0-9_.-]+$")

    @model_validator(mode="after")
    def validate_tls(self) -> "ValkeyConfig":
        if self.insecure_tls and not self.ssl:
            msg = "insecure_tls requires ssl=True"
            raise ValueError(msg)
        return self

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value() if self.password is not None else None,
            "ssl": self.ssl,
            "insecure_tls": self.insecure_tls,
            "cmd": self.cmd,
            "request_timeout_ms": self.request_timeout_ms,
            "connection_timeout_ms": self.connection_timeout_ms,
            "collection_name": self.collection_name,
        }


class ValkeyIndexConfig(BaseModel):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type in {None, MetricType.COSINE}:
            return "COSINE"
        if self.metric_type in {MetricType.L2, MetricType.IP}:
            return self.metric_type.value
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)


class ValkeyHNSWConfig(ValkeyIndexConfig, DBCaseConfig):
    M: int = Field(default=16, gt=0)
    efConstruction: int = Field(default=200, gt=0)
    ef: int = Field(default=10, gt=0)
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }
