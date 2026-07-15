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


# Infino's BM25 k1/b are compile-time constants and its analyzer is not
# configurable from the binding, so the FTS index has no tunable params.
_INFINO_BM25_K1 = 1.2
_INFINO_BM25_B = 0.75


class InfinoFTSConfig(BaseModel, DBCaseConfig):
    """Marks a run as full-text (BM25) rather than vector."""

    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}

    def apply_fts_manifest(self, bm25_params: dict[str, float], analyzer_params: dict) -> tuple[DBCaseConfig, dict]:
        fixed = {"k1": _INFINO_BM25_K1, "b": _INFINO_BM25_B}
        applied = {k: v for k, v in bm25_params.items() if k in fixed and v == fixed[k]}
        return self, {
            "applied_bm25_params": applied,
            "unapplied_bm25_params": {k: v for k, v in bm25_params.items() if k not in applied},
            "applied_analyzer_params": {},
            "unapplied_analyzer_params": dict(analyzer_params),
        }
