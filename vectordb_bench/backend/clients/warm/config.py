import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType

DEFAULT_WARM_PATH = str(Path(tempfile.gettempdir()) / "vectordb_bench" / "warm")


class WarmConfig(DBConfig):
    """WARM is embedded: `path` is the directory that holds the index files."""

    db_label: str
    path: str = DEFAULT_WARM_PATH

    def to_dict(self) -> dict:
        return {"path": self.path}


class WarmIndexConfig(BaseModel, DBCaseConfig):
    """WARM graph index parameters.

    `m` is the HNSW M of the shared board configuration; WARM's `degree` is the out-degree of its
    single-layer graph, which matches HNSW's layer-0 degree 2*M. `ef_construction` is `build_ef`.
    `ef_search` is the candidate pool size and `rerank` how many pool candidates are re-scored with
    the residual code; 0 selects `rerank = max(ef, k)`, re-scoring the whole pool. Recall only reaches
    its ceiling when rerank covers the pool, so the sweep parameter is `ef_search` alone.
    """

    metric_type: MetricType | None = None
    m: int = 16
    ef_construction: int = 300
    ef_search: int = 128
    rerank: int = 0
    code_bits: int = 4
    residual_bits: int = 8
    threads: int = 0

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "ip"
        if self.metric_type in (MetricType.COSINE, None):
            return "angular"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "degree": 2 * self.m,
            "build_ef": self.ef_construction,
            "code_bits": self.code_bits,
            "residual_bits": self.residual_bits,
            "threads": self.threads,
        }

    def search_param(self) -> dict:
        return {"ef": self.ef_search, "rerank": self.rerank}
