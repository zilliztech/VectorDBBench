"""VectorDBBench client for the embedded indexes of RaBitQ-Library (``rabitqlib``).

The library runs in-process, while VectorDBBench loads, optimizes and searches in
separate processes, so all state lives in files under ``db_config["path"]``:

- load: ``insert_embeddings`` appends each batch to a raw buffer
  (``buffer.vectors.f32`` and ``buffer.ids.i64``, row-aligned under one lock,
  since up to four insert threads share the client);
- optimize: the index type's config builds the buffer into one index in a single
  ``add``, which also trains any centroids on the whole dataset. The client saves
  it with the map from library point IDs to dataset IDs and a manifest of the
  build parameters, all named after the index type (``rabitqlib_hnsw.*``), so
  index types can share a path;
- search: every search process loads the saved index when it receives the
  client, before the concurrent-search timer starts.

Only unfiltered static performance cases are supported: the indexes cannot
filter on scalars, and vectors arriving after the build are rejected rather
than lost.
"""

from __future__ import annotations

import array
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rabitqlib

from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import NonRetryableInsertError, VectorDB

if TYPE_CHECKING:
    from .config import RaBitQLibIndexConfig

log = logging.getLogger(__name__)

# Library point ID for an empty result slot (k above the number of points).
MISSING_ID = np.iinfo(np.uint32).max
# The rabitqlib indexes reject smaller dimensions.
MIN_DIM = 64
MANIFEST_VERSION = 1

VECTORS_FILE = "buffer.vectors.f32"
IDS_FILE = "buffer.ids.i64"
BUFFER_FILES = (VECTORS_FILE, IDS_FILE)


class RaBitQLib(VectorDB):
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    supports_batch_search: bool = True
    # Insert threads share one client; appends are serialized by self._lock.
    thread_safe: bool = True

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: RaBitQLibIndexConfig,
        collection_name: str = "",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "RaBitQLib"
        if dim < MIN_DIM:
            msg = f"{self.name} requires dim >= {MIN_DIM}, got {dim}"
            raise ValueError(msg)
        self.dim = dim
        self.case_config = db_case_config
        self.index_type = db_case_config.index.value
        self.index_param = db_case_config.index_param()
        self.path = Path(db_config["path"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = f"{self.index_type}.index"
        self.id_map_file = f"{self.index_type}.id_map.npy"
        self.manifest_file = f"{self.index_type}.manifest.json"

        # Runtime state, created per process and never pickled.
        self._lock: threading.Lock | None = None
        self._writers: tuple | None = None
        self._index: Any = None
        self._id_map: np.ndarray | None = None
        self._search_kwargs: dict[int, dict] = {}

        if drop_old:
            # Remove only the buffer and this index type's files: the path may hold other data.
            for name in (*BUFFER_FILES, self.index_file, self.id_map_file, self.manifest_file):
                self._file(name).unlink(missing_ok=True)
            log.info(f"{self.name}: cleared {self.index_type} files under {self.path}")
        elif self._file(self.manifest_file).exists():
            self._check_manifest()
        elif any(self._file(name).exists() for name in BUFFER_FILES):
            msg = (
                f"{self.name}: {self.path} holds an unbuilt load buffer from an earlier run; "
                "rerun with drop_old so it is not loaded twice"
            )
            raise RuntimeError(msg)

    def _file(self, name: str) -> Path:
        return self.path / name

    def _built(self) -> bool:
        return self._file(self.manifest_file).exists()

    def __getstate__(self) -> dict:
        """Drop per-process handles: the client is pickled into spawned processes."""
        state = self.__dict__.copy()
        for key in ("_lock", "_writers", "_index", "_id_map"):
            state[key] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Load a built index as soon as a spawned process receives the client.

        The concurrent search runner calls ``init`` only after its start barrier, so
        loading there would count every process's index load (seconds per GB, all
        processes reading at once) as search time and understate QPS. Unpickling
        happens before the barrier. Load and optimize processes find no built index
        and skip this.
        """
        self.__dict__.update(state)
        if self._built():
            self._load_index()

    def _load_index(self) -> None:
        self._index = self.case_config.load(str(self._file(self.index_file)))
        self._id_map = np.load(self._file(self.id_map_file))

    def _check_manifest(self) -> None:
        """Refuse to reuse an index built with other parameters or data."""
        manifest = json.loads(self._file(self.manifest_file).read_text())
        if manifest.get("dim") != self.dim or manifest.get("index_param") != self.index_param:
            msg = (
                f"{self.name}: the {self.index_type} index under {self.path} was built with "
                f"dim={manifest.get('dim')}, {manifest.get('index_param')}; "
                f"this case needs dim={self.dim}, {self.index_param}. Rerun with drop_old to rebuild."
            )
            raise ValueError(msg)
        log.info(f"{self.name}: reusing {self.index_type} index under {self.path} ({manifest.get('count')} vectors)")

    @contextmanager
    def init(self):
        self._lock = threading.Lock()
        if self._index is None and self._built():
            self._load_index()
        try:
            yield
        finally:
            self._close_writers()
            self._index = None
            self._id_map = None
            self._lock = None

    def _close_writers(self) -> None:
        if self._writers is not None:
            for f in self._writers:
                f.close()
            self._writers = None

    def need_normalize_cosine(self) -> bool:
        # The library has no cosine metric: cosine runs as IP on unit vectors.
        return True

    def prepare_filter(self, filters: Filter):
        if filters.type != FilterOp.NonFilter:
            msg = f"{self.name} does not support filtered search ({filters.type})"
            raise ValueError(msg)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        if self._built():
            msg = (
                f"{self.name} builds its index once, in optimize; vectors inserted after the build "
                "are not supported (static performance cases only)"
            )
            return 0, NonRetryableInsertError(msg)
        try:
            vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
            ids = np.asarray(metadata, dtype=np.int64)
        except ValueError as e:
            return 0, NonRetryableInsertError(f"{self.name}: batch is not a float matrix: {e}")
        if vectors.ndim != 2 or vectors.shape != (len(ids), self.dim):
            msg = f"{self.name}: expected {len(ids)} vectors of dim {self.dim}, got shape {vectors.shape}"
            return 0, NonRetryableInsertError(msg)
        try:
            with self._lock:
                if self._writers is None:
                    self._writers = (
                        self._file(VECTORS_FILE).open("ab"),
                        self._file(IDS_FILE).open("ab"),
                    )
                vectors_out, ids_out = self._writers
                vectors_out.write(vectors.tobytes())
                ids_out.write(ids.tobytes())
        except Exception as e:
            # A failed write may have appended part of a batch; retrying would duplicate it.
            log.warning(f"{self.name}: failed to buffer {len(metadata)} vectors: {e}")
            return 0, NonRetryableInsertError(str(e))
        return len(ids), None

    def optimize(self, data_size: int | None = None):
        self._close_writers()
        if self._built():
            log.info(f"{self.name}: {self.index_type} index already built under {self.path}")
            return

        ids = np.fromfile(self._file(IDS_FILE), dtype=np.int64)
        n = len(ids)
        vectors_bytes = self._file(VECTORS_FILE).stat().st_size
        if n == 0 or vectors_bytes != n * self.dim * 4:
            msg = f"{self.name}: load buffer holds {n} ids but {vectors_bytes} vector bytes (dim {self.dim})"
            raise RuntimeError(msg)
        if len(np.unique(ids)) != n:
            msg = f"{self.name}: load buffer holds duplicate ids"
            raise RuntimeError(msg)
        vectors = np.memmap(self._file(VECTORS_FILE), dtype=np.float32, mode="r", shape=(n, self.dim))

        build_threads = self.case_config.build_threads or None
        log.info(
            f"{self.name}: building {self.index_type} over {n} vectors, {self.index_param}, "
            f"build_threads={build_threads or 'all'}"
        )
        start = time.perf_counter()
        index, point_ids = self.case_config.build(self.dim, vectors)
        build_seconds = time.perf_counter() - start

        # Library point IDs index id_map; each maps to the dataset ID of its row.
        id_map = np.empty(n, dtype=np.int64)
        id_map[point_ids] = ids
        index.save(str(self._file(self.index_file)))
        np.save(self._file(self.id_map_file), id_map)
        index_bytes = self._file(self.index_file).stat().st_size
        manifest = {
            "version": MANIFEST_VERSION,
            "rabitqlib_version": rabitqlib.__version__,
            "index_type": self.index_type,
            "dim": self.dim,
            "count": n,
            "index_param": self.index_param,
            "build_threads": build_threads or os.cpu_count(),
            "build_seconds": round(build_seconds, 3),
            "index_bytes": index_bytes,
        }
        # Written last: its presence marks a complete build.
        self._file(self.manifest_file).write_text(json.dumps(manifest, indent=2))
        for name in BUFFER_FILES:
            self._file(name).unlink(missing_ok=True)
        log.info(
            f"{self.name}: built {self.index_type} in {build_seconds:.1f}s; "
            f"index file {index_bytes / 2**20:.1f} MiB, loaded once per search process"
        )

    def _require_index(self) -> Any:
        if self._index is None:
            msg = f"{self.name}: no built {self.index_type} index under {self.path}; run the load stage first"
            raise RuntimeError(msg)
        return self._index

    def _kwargs_for(self, k: int) -> dict:
        kwargs = self._search_kwargs.get(k)
        if kwargs is None:
            kwargs = self._search_kwargs[k] = {**self.case_config.search_kwargs(k), "num_threads": 1}
        return kwargs

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
        **kwargs,
    ) -> list[int]:
        index = self._require_index()
        # array.array converts a list of floats to float32 faster than np.asarray,
        # which shows at sub-millisecond latencies.
        point_ids, _ = index.search(
            np.frombuffer(array.array("f", query), dtype=np.float32).reshape(1, -1),
            k,
            **self._kwargs_for(k),
        )
        row = point_ids[0]
        return self._id_map[row[row != MISSING_ID]].tolist()

    def search_embeddings(
        self,
        queries: list[list[float]],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[list[int]]:
        index = self._require_index()
        point_ids, _ = index.search(np.asarray(queries, dtype=np.float32), k, **self._kwargs_for(k))
        return [self._id_map[row[row != MISSING_ID]].tolist() for row in point_ids]
