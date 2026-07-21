"""VectorDBBench connector for Telys (the Algenta-powered, on-device vector-execution layer).

Telys accelerates FILTERED search by physical layout: the partition key becomes a contiguous slice,
so `label == X` reads one block instead of scanning + masking the whole table. This connector therefore
partitions by the label field for the label-filter case (Telys's wedge); the no-filter case uses a single
constant partition (plain ANN over one block). It talks to a running `telys serve` over TCP via the public
`telys.client` (RemoteTelys) — a fair server-vs-server measurement.

Prereqs:
  pip install telys                      # the public SDK (RemoteTelys client)
  telys serve --path ./telys-vdbbench --host 0.0.0.0 --port 9099 --access-token "$TELYS_ACCESS_TOKEN"

Metric note: Telys scores raw-vector collections by inner product; L2-normalized IP == cosine, so
`need_normalize_cosine()` is True and this connector targets cosine/IP datasets. (Pure-L2 datasets are
not matched by this v1 — Telys raw collections do not expose an L2 metric knob.)
"""

import logging
from contextlib import contextmanager

import numpy as np

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import TelysIndexConfig

log = logging.getLogger(__name__)

LABEL_FIELD = "labels"  # metadata field the label-filter case filters on (e.g. labels == "label_5p")
SCOPE_FIELD = "scope"   # single constant partition key for the no-filter case


class TelysClient(VectorDB):
    """Telys over `telys serve` (TCP). Label-filter equality maps to Telys's single-key partition wedge."""

    # Equality label filter == the contiguous-slice wedge; int-range (NumGE) is intentionally NOT declared
    # (Telys's Eq is equality, not a range — a range filter would be a scan, not the wedge).
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.StrEqual]
    name: str = "Telys"
    # Each concurrent runner gets its OWN RemoteTelys socket (a connection serializes one in-flight request),
    # so concurrency comes from many connections, not one shared handle.
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TelysIndexConfig | None,
        collection_name: str = "VectorDBBenchTelys",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = int(dim)
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.with_scalar_labels = with_scalar_labels
        self.partition_by = LABEL_FIELD if with_scalar_labels else SCOPE_FIELD
        self.filter_columns = [LABEL_FIELD] if with_scalar_labels else []
        self._where: dict | None = None  # active single-key filter, set by prepare_filter()
        self._eng = None
        self._col = None

        eng = self._connect()
        try:
            if drop_old:
                try:
                    eng.drop_collection(collection_name)
                    log.info("Telys: dropped old collection %s", collection_name)
                except Exception as e:  # noqa: BLE001
                    log.warning("Telys drop_old (%s): %s", collection_name, e)
            eng.create_collection(
                collection_name, self.dim, self.partition_by,
                filter_columns=self.filter_columns, dtype="f32", embedder=None,
            )
        finally:
            eng.close()

    def _connect(self):
        from telys.client import connect

        c = self.db_config
        return connect(f"tcp://{c['host']}:{c['port']}", token=c.get("access_token"))

    @contextmanager
    def init(self):
        self._eng = self._connect()
        self._col = self._eng.open_collection(self.collection_name)
        try:
            yield
        finally:
            try:
                self._eng.close()
            finally:
                self._eng = None
                self._col = None

    def need_normalize_cosine(self) -> bool:
        return True  # Telys raw collections score by IP; L2-normalized IP == cosine

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self._col is not None, "call init() first"
        try:
            ids = [int(m) for m in metadata]
            if self.with_scalar_labels:
                assert labels_data is not None, "label-filter case requires labels_data"
                md = [{LABEL_FIELD: labels_data[i]} for i in range(len(ids))]
            else:
                md = [{SCOPE_FIELD: "all"} for _ in ids]
            vecs = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
            self._col.add(vecs, ids, md)
            return len(ids), None
        except Exception as e:  # noqa: BLE001
            log.warning("Telys insert failed: %s", e)
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self._where = None
        elif filters.type == FilterOp.StrEqual:
            # single-key equality -> Telys reads one contiguous partition slice (the wedge)
            self._where = {LABEL_FIELD: filters.label_value}
        else:
            msg = f"Telys connector does not support filter type {filters.type}"
            raise RuntimeError(msg)

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        assert self._col is not None, "call init() first"
        res = self._col.search(np.asarray(query, dtype=np.float32).reshape(-1), top_k=k, where=self._where)
        return [int(i) for i in res["ids"]]

    def optimize(self, data_size: int | None = None):
        """Build per-partition IVF for oversized partitions, then persist (the commit point)."""
        p = self.case_config.index_param() if self.case_config else {}
        col = self._col
        eng = None
        if col is None:  # optimize may run outside an init() scope in some harness paths
            eng = self._connect()
            col = eng.open_collection(self.collection_name)
        try:
            col.build_ivf(min_rows=int(p.get("min_rows", 20000)), target_recall=float(p.get("target_recall", 0.98)))
            col.save()
        finally:
            if eng is not None:
                eng.close()
