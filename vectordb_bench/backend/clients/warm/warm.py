"""WARM (WritersLogic Adaptive Retention & Memory) vector index, embedded in the benchmark process.

The harness inserts in one process, optimizes in another and searches in a pool of spawned
processes, so state crosses process boundaries through files in `path/<collection>`: inserts write
vector chunks, `optimize` fits the index once and saves it, and every search process loads the saved
index. WARM returns the row number of each neighbour in the fitted matrix; rows are fitted in
ascending id order and `ids.npy` maps a row back to the dataset id.
"""

import json
import logging
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..api import DBCaseConfig, VectorDB
from .config import WarmIndexConfig

log = logging.getLogger(__name__)

INDEX_FILE = "index.warm"
IDS_FILE = "ids.npy"
CHUNK_DIR = "chunks"
MANIFEST_FILE = "manifest.json"


class Warm(VectorDB):
    # VectorIndex.query takes &mut self, so a live index belongs to one thread.
    thread_safe: bool = False
    supports_batch_search: bool = True

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        dataset_identity: dict | None = None,
        **kwargs,
    ):
        if not isinstance(dataset_identity, dict) or not dataset_identity:
            raise ValueError("WARM requires a dataset identity to validate saved-index reuse")
        self.name = "Warm"
        self.dim = dim
        self.case_config = db_case_config if db_case_config is not None else WarmIndexConfig()
        self.root = Path(db_config["path"]) / collection_name
        self._index = None
        self._ids = None
        self._k = None
        self._identity = json.loads(
            json.dumps(
                {
                    "version": 1,
                    "dataset": dataset_identity,
                    "dim": dim,
                    "build": self.case_config.index_param(),
                }
            )
        )

        if drop_old and self.root.exists():
            shutil.rmtree(self.root)
        if (self.root / MANIFEST_FILE).exists():
            self._read_manifest()
        elif self.root.exists() and any(self.root.iterdir()):
            raise ValueError("WARM index has no identity manifest; rebuild without --skip-drop-old")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_manifest(built=False)
        (self.root / CHUNK_DIR).mkdir(parents=True, exist_ok=True)

    def _read_manifest(self) -> dict:
        try:
            manifest = json.loads((self.root / MANIFEST_FILE).read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("WARM identity manifest is unreadable; rebuild without --skip-drop-old") from exc
        if not isinstance(manifest, dict) or manifest.get("identity") != self._identity:
            raise ValueError("WARM dataset or build settings do not match; rebuild without --skip-drop-old")
        built = manifest.get("built")
        files_present = [(self.root / name).is_file() for name in (INDEX_FILE, IDS_FILE)]
        if type(built) is not bool or any(present != built for present in files_present):
            raise ValueError("WARM saved index is incomplete; rebuild without --skip-drop-old")
        return manifest

    def _write_manifest(self, *, built: bool) -> None:
        path = self.root / f"{uuid.uuid4().hex}.manifest.tmp"
        try:
            path.write_text(json.dumps({"identity": self._identity, "built": built}, sort_keys=True) + "\n")
            path.replace(self.root / MANIFEST_FILE)
        finally:
            path.unlink(missing_ok=True)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_index"] = None
        state["_ids"] = None
        state["_k"] = None
        return state

    def _new_index(self):
        import writerslogic_warm

        return writerslogic_warm.VectorIndex(**self.case_config.index_param())

    @contextmanager
    def init(self):
        try:
            if self._read_manifest()["built"]:
                self._index = self._new_index()
                self._index.load(self.root / INDEX_FILE)
                self._ids = np.load(self.root / IDS_FILE, allow_pickle=False)
            self._k = None
            yield
        finally:
            self._index = None
            self._ids = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            if self._read_manifest()["built"]:
                msg = "WARM cannot insert into a saved index; rebuild without --skip-drop-old"
                raise ValueError(msg)  # noqa: TRY301
            vectors = np.asarray(embeddings, dtype=np.float32)
            ids = np.asarray(metadata, dtype=np.int64)
            if vectors.ndim != 2 or vectors.shape != (len(ids), self.dim):
                msg = f"expected ({len(ids)}, {self.dim}) embeddings, got {vectors.shape}"
                raise ValueError(msg)  # noqa: TRY301
            name = uuid.uuid4().hex
            chunk_dir = self.root / CHUNK_DIR
            # .tmp files are ignored by optimize; the rename publishes a complete pair.
            np.save(chunk_dir / f"{name}.vec.tmp.npy", vectors)
            np.save(chunk_dir / f"{name}.ids.tmp.npy", ids)
            (chunk_dir / f"{name}.vec.tmp.npy").replace(chunk_dir / f"{name}.vec.npy")
            (chunk_dir / f"{name}.ids.tmp.npy").replace(chunk_dir / f"{name}.ids.npy")
        except Exception as e:
            log.warning(f"Failed to insert data into WARM chunk store: {e}")
            return 0, e
        return len(ids), None

    def optimize(self, data_size: int | None = None):
        manifest = self._read_manifest()
        chunk_dir = self.root / CHUNK_DIR
        names = sorted(p.name[: -len(".ids.npy")] for p in chunk_dir.glob("*.ids.npy") if ".tmp." not in p.name)
        if not names:
            if manifest["built"]:
                return
            msg = "no inserted vectors to build the WARM index from"
            raise RuntimeError(msg)

        ids = np.concatenate([np.load(chunk_dir / f"{n}.ids.npy") for n in names])
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        if len(sorted_ids) > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
            msg = "duplicate ids inserted"
            raise ValueError(msg)

        matrix = np.empty((len(ids), self.dim), dtype=np.float32)
        row = 0
        for n in names:
            part = np.load(chunk_dir / f"{n}.vec.npy", mmap_mode="r")
            matrix[row : row + len(part)] = part
            row += len(part)
        # Fit in id order so the row number is the rank of the id.
        matrix = matrix[order]

        index = self._new_index()
        index.fit(matrix)
        del matrix
        index.save(self.root / INDEX_FILE)
        np.save(self.root / IDS_FILE, sorted_ids)
        # Publish readiness only after both files are complete; interrupted builds fail closed.
        self._write_manifest(built=True)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        log.info(f"WARM index built over {len(ids)} vectors: {index.index_bytes} payload bytes")

    def _set_search_args(self, k: int) -> None:
        if self._index is None:
            raise RuntimeError("WARM index is not built; run the load stage before searching")
        if self._k != k:
            params = self.case_config.search_param()
            self._index.set_query_arguments(ef=params["ef"], rerank=params["rerank"] or max(params["ef"], k))
            self._k = k

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        self._set_search_args(k)
        rows = self._index.query(np.asarray(query, dtype=np.float32), k)
        return self._ids[rows].tolist()

    def search_embeddings(self, queries: list[list[float]], k: int = 100, **kwargs) -> list[list[int]]:
        self._set_search_args(k)
        batch = np.asarray(queries, dtype=np.float32)
        if batch.ndim != 2 or batch.shape != (len(queries), self.dim):
            msg = f"expected ({len(queries)}, {self.dim}) queries, got {batch.shape}"
            raise ValueError(msg)
        rows = self._index.batch_query_array(batch, k)
        if rows.shape[0] != len(queries):
            msg = f"WARM returned {rows.shape[0]} result rows for {len(queries)} queries"
            raise RuntimeError(msg)
        # Rows are padded with -1 when the index holds fewer than k vectors.
        return [self._ids[row[row >= 0]].tolist() for row in rows]
