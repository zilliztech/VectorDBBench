from pydantic import BaseModel

from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, MetricType

# Infino distance metrics; all are distances where smaller means nearer.
_METRIC_MAP = {
    MetricType.COSINE: "cosine",
    MetricType.L2: "l2sq",
    MetricType.IP: "negdot",
}


class InfinoConfig(DBConfig):
    data_path: str = "/tmp/vectordb_bench/infino"
    table_name: str = "vdbbench_infino"

    def to_dict(self) -> dict:
        return {"data_path": self.data_path, "table_name": self.table_name}


class InfinoIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    n_cent: int = 256
    nprobe: int = 32

    def parse_metric(self) -> str:
        if self.metric_type not in _METRIC_MAP:
            msg = f"Infino does not support metric {self.metric_type}"
            raise ValueError(msg)
        return _METRIC_MAP[self.metric_type]

    def index_param(self) -> dict:
        return {"metric": self.parse_metric(), "n_cent": self.n_cent}

    def search_param(self) -> dict:
        return {"nprobe": self.nprobe}
