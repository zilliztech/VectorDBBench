from pydantic import BaseModel

from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, MetricType

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
    n_cent: int = 256
    # Unset => the engine picks. The engine owns these defaults and its
    # recall numbers are measured on that path, so the client must not
    # substitute values of its own; only forward what a caller asked for.
    nprobe: int | None = None
    rerank_mult: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type not in _METRIC_MAP:
            msg = f"Infino does not support metric {self.metric_type}"
            raise ValueError(msg)
        return _METRIC_MAP[self.metric_type]

    def index_param(self) -> dict:
        return {"metric": self.parse_metric(), "n_cent": self.n_cent}

    def search_param(self) -> dict:
        return {
            k: v
            for k, v in (("nprobe", self.nprobe), ("rerank_mult", self.rerank_mult))
            if v is not None
        }


# Infino's BM25 k1/b are compile-time constants. The analyzer (tokenizer)
# IS selectable from the binding: "ascii_lower" (default) or "standard"
# (Unicode UAX #29 + lowercase, keeps non-ASCII).
_INFINO_BM25_K1 = 1.2
_INFINO_BM25_B = 0.75

# GT analyzer specs (tokenizer + filter set) that a given infino analyzer
# reproduces exactly. The FTS math GT is built with the "standard" tokenizer
# + a lowercase filter; infino's "standard" tokenizer already lowercases, so
# it reproduces that spec.
_INFINO_ANALYZER_FOR = {
    ("standard", frozenset({"lowercase"})): "standard",
    ("standard", frozenset()): "standard",
}


class InfinoFTSConfig(BaseModel, DBCaseConfig):
    """Marks a run as full-text (BM25) rather than vector."""

    metric_type: MetricType | None = None
    # Infino tokenizer to build + query the text column with. Set from the
    # GT manifest by `apply_fts_manifest`; defaults to the ASCII tokenizer.
    analyzer: str = "ascii_lower"

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}

    def apply_fts_manifest(self, bm25_params: dict[str, float], analyzer_params: dict) -> tuple[DBCaseConfig, dict]:
        fixed = {"k1": _INFINO_BM25_K1, "b": _INFINO_BM25_B}
        applied = {k: v for k, v in bm25_params.items() if k in fixed and v == fixed[k]}

        # Select the infino tokenizer that reproduces the GT analyzer. If we
        # can match it exactly, build + query with it (recall then reflects
        # ranking parity, not tokenization drift); otherwise keep the ASCII
        # default and report the analyzer as unapplied.
        params = analyzer_params or {}
        key = (params.get("tokenizer"), frozenset(params.get("filter", [])))
        infino_analyzer = _INFINO_ANALYZER_FOR.get(key)
        if infino_analyzer is not None:
            self.analyzer = infino_analyzer
            applied_analyzer = dict(analyzer_params)
            unapplied_analyzer: dict = {}
        else:
            applied_analyzer = {}
            unapplied_analyzer = dict(analyzer_params)

        return self, {
            "applied_bm25_params": applied,
            "unapplied_bm25_params": {k: v for k, v in bm25_params.items() if k not in applied},
            "applied_analyzer_params": applied_analyzer,
            "unapplied_analyzer_params": unapplied_analyzer,
        }
