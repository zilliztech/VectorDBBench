from pydantic import BaseModel, field_validator

from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, MetricType

# Vector serving path, bridged to the engine's `vector.search_mode` config key
# by the client before connect (see infino.py) — deliberately a config override,
# not a public engine-API field, so nothing goes vestigial if the engine default
# changes later. "ivf" (default) serves the reclaimable IVF scan; "hnsw_ivf"
# builds + serves a resident HNSW graph over the Sq16 vectors (automatic ivf
# fallback).
_SEARCH_MODES = ("ivf", "hnsw_ivf")

# Infino distance metrics; all are distances where smaller means nearer.
_METRIC_MAP = {
    MetricType.COSINE: "cosine",
    MetricType.L2: "l2sq",
    MetricType.IP: "negdot",
}


# Disk-cache ceiling, not a preallocation: sized well above the 10 GiB engine
# default so large corpora stay cached instead of falling back to range-only reads.
_DEFAULT_CACHE_BUDGET_BYTES = 64 * 1024**3


class InfinoConfig(DBConfig):
    data_path: str = "/tmp/vectordb_bench/infino"
    table_name: str = "vdbbench_infino"
    cache_budget_bytes: int = _DEFAULT_CACHE_BUDGET_BYTES
    cache_dir: str | None = None
    storage_options: dict[str, str] | None = None

    def to_dict(self) -> dict:
        return {
            "data_path": self.data_path,
            "table_name": self.table_name,
            "cache_budget_bytes": self.cache_budget_bytes,
            "cache_dir": self.cache_dir,
            "storage_options": self.storage_options,
        }


class InfinoIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    # Probe width and rerank budget stay engine-decided (calibrated per table at
    # optimize time); only the serving-path selector is forwarded, and it goes
    # through the engine config file, not IndexSpec (see _SEARCH_MODES above).
    # Default hnsw_ivf: the resident-HNSW path is what this client benchmarks.
    search_mode: str = "hnsw_ivf"
    # Serve-time HNSW beam for search_mode=hnsw_ivf, bridged to the engine's
    # vector.hnsw_ef_search config key (see infino.py). 0 (default) serves each
    # query at the graph's stamped k->ef curve; a positive value overrides it
    # with a fixed beam, so sweeping ef across runs traces the recall/latency
    # curve of ONE built graph with no rebuild — the direct analog of how the
    # curated leaderboard sweeps ZillizCloud's per-run `level`.
    ef: int = 0

    @field_validator("search_mode")
    @classmethod
    def _validate_search_mode(cls, v: str) -> str:
        if v not in _SEARCH_MODES:
            msg = f"Infino search_mode must be one of {_SEARCH_MODES}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("ef")
    @classmethod
    def _validate_ef(cls, v: int) -> int:
        if v < 0:
            msg = f"Infino ef must be >= 0 (0 = use the stamped curve), got {v}"
            raise ValueError(msg)
        return v

    def parse_metric(self) -> str:
        if self.metric_type not in _METRIC_MAP:
            msg = f"Infino does not support metric {self.metric_type}"
            raise ValueError(msg)
        return _METRIC_MAP[self.metric_type]

    def index_param(self) -> dict:
        return {"metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {}
