from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class KiviDBConfig(DBConfig):
    host: SecretStr
    port: int = 6380
    password: SecretStr | None = None
    ssl: bool = False

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value() if self.password else None,
            "ssl": self.ssl,
        }


class KiviDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "L2"
        if self.metric_type == MetricType.IP:
            return "IP"
        return "COSINE"


class KiviDBHNSWConfig(KiviDBIndexConfig):
    M: int = 16
    ef_construction: int = 200
    ef_runtime: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "index_type": self.index.value,
            "metric": self.parse_metric(),
            "m": self.M,
            "ef_construction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {"ef_runtime": self.ef_runtime}


class KiviDBFLATConfig(KiviDBIndexConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {"index_type": self.index.value, "metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {"ef_runtime": None}


_kividb_case_config = {
    IndexType.HNSW: KiviDBHNSWConfig,
    IndexType.Flat: KiviDBFLATConfig,
}
