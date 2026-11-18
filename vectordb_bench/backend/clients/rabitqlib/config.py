"""Settings for the embedded indexes of RaBitQ-Library (``rabitqlib``).

Each index type is one case config: its build parameters, how to build and load
the index, and its search arguments. Everything else (buffering, saving, loading
in search processes, ID mapping) is shared by the client.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

if TYPE_CHECKING:
    import numpy as np


class RaBitQLibConfig(DBConfig):
    """Where the embedded indexes keep their files.

    There is no server: the client writes the load buffer, the built index and
    its manifest under ``path``. Different index types can share one path.
    """

    path: str

    def to_dict(self) -> dict:
        return {"path": self.path}


class RaBitQLibIndexConfig(BaseModel, DBCaseConfig):
    """Shared settings of every rabitqlib index type.

    ``build_threads=0`` uses every available CPU, capped by ``OMP_NUM_THREADS``.

    Unlike most case configs, each index type also builds, loads and searches its
    library index here, so adding an index type does not grow the shared client.
    The class stays abstract: ``index_param`` and ``search_param`` come from
    ``DBCaseConfig``.
    """

    index: IndexType
    metric_type: MetricType | None = None
    seed: int = 100
    build_threads: int = 0

    def library_metric(self) -> str:
        """The rabitqlib metric for the dataset's metric.

        Cosine runs as inner product on vectors VectorDBBench has normalized
        (the client returns True from ``need_normalize_cosine``).
        """
        if self.metric_type in (MetricType.COSINE, MetricType.IP):
            return "ip"
        if self.metric_type == MetricType.L2:
            return "l2"
        msg = f"rabitqlib indexes support L2, IP and COSINE datasets, not {self.metric_type}"
        raise ValueError(msg)

    @abstractmethod
    def build(
        self, dim: int, vectors: np.ndarray, index_param: dict, num_threads: int | None
    ) -> tuple[Any, np.ndarray]:
        """Build the index over all vectors from ``index_param()``; return it and each row's point ID."""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load an index saved by the client."""

    @abstractmethod
    def search_kwargs(self, k: int) -> dict:
        """Keyword arguments of the index's ``search`` besides queries and k, from ``search_param()``."""


class RaBitQLibHNSWConfig(RaBitQLibIndexConfig):
    """HNSW-RaBitQ: an HNSW graph searched over RaBitQ codes.

    ``num_clusters`` centroids are trained by the library on the first ``add``,
    with k-means on a sample of up to 256 x ``num_clusters`` vectors (the
    library's defaults); every vector is then encoded against its nearest
    centroid. ``nbits=9`` (sign bit + 8) is the highest precision; the library
    defaults to 7.
    """

    index: IndexType = IndexType.RABITQLIB_HNSW
    nbits: int = 9
    num_clusters: int = 16
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 128

    def index_param(self) -> dict:
        return {
            "metric": self.library_metric(),
            "nbits": self.nbits,
            "num_clusters": self.num_clusters,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "seed": self.seed,
        }

    def search_param(self) -> dict:
        return {"ef": self.ef_search}

    def build(
        self, dim: int, vectors: np.ndarray, index_param: dict, num_threads: int | None
    ) -> tuple[Any, np.ndarray]:
        import rabitqlib

        index = rabitqlib.HnswIndex(dim, **index_param)
        point_ids = index.add(vectors, num_threads=num_threads)
        return index, point_ids

    def load(self, path: str) -> Any:
        import rabitqlib

        return rabitqlib.HnswIndex.load(path)

    def search_kwargs(self, k: int) -> dict:
        # HNSW needs a candidate list of at least k.
        return {"ef": max(self.search_param()["ef"], k)}


_rabitqlib_case_config: dict[IndexType, type[RaBitQLibIndexConfig]] = {
    IndexType.RABITQLIB_HNSW: RaBitQLibHNSWConfig,
}
