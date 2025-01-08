from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class MemoryDBConfig(DBConfig):
    host: SecretStr
    password: SecretStr | None = None
    port: int | None = None
    ssl: bool | None = None
    cmd: bool | None = None
    ssl_ca_certs: str | None = None

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value() if self.password else None,
            "ssl": self.ssl,
            "cmd": self.cmd,
            "ssl_ca_certs": self.ssl_ca_certs,
        }


class MemoryDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    insert_batch_size: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.IP:
            return "ip"
        return "cosine"


class MemoryDBHNSWConfig(MemoryDBIndexConfig):
    M: int | None = 16
    ef_construction: int | None = 64
    ef_runtime: int | None = 10
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "m": self.M,
            "ef_construction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {
            "ef_runtime": self.ef_runtime,
        }
