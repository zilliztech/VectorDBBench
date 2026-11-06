"""WARM (WritersLogic Adaptive Retention & Memory) vector index, embedded in the benchmark process.

The harness inserts in one process, optimizes in another and searches in a pool of spawned
processes, so state crosses process boundaries through files in `path/<collection>`: inserts write
vector chunks, `optimize` fits the index once and saves it, and every search process loads the saved
index. WARM returns the row number of each neighbour in the fitted matrix; rows are fitted in
ascending id order and `ids.npy` maps a row back to the dataset id.
"""

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
        **kwargs,
    ):
        self.name = "Warm"
        self.dim = dim
        self.case_config = db_case_config if db_case_config is not None else WarmIndexConfig()
        self.root = Path(db_config["path"]) / collection_name
        self._index = None
        self._ids = None
        self._k = None

        if drop_old:
            shutil.rmtree(self.root, ignore_errors=True)
        (self.root / CHUNK_DIR).mkdir(parents=True, exist_ok=True)

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
        index_path = self.root / INDEX_FILE
        if index_path.exists():
            self._index = self._new_index()
            self._index.load(index_path)
            self._ids = np.load(self.root / IDS_FILE)
        self._k = None
        try:
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
        chunk_dir = self.root / CHUNK_DIR
        names = sorted(p.name[: -len(".ids.npy")] for p in chunk_dir.glob("*.ids.npy") if ".tmp." not in p.name)
        if not names:
            if (self.root / INDEX_FILE).exists():
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
        shutil.rmtree(chunk_dir, ignore_errors=True)
        log.info(f"WARM index built over {len(ids)} vectors: {index.index_bytes} payload bytes")

    def _set_search_args(self, k: int) -> None:
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
