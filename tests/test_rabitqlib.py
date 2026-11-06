"""Unit tests for the embedded rabitqlib client (HNSW-RaBitQ index).

They build small real indexes with rabitqlib (no server) and cover:
  - metric mapping and registration (init_cls / config_cls / case_config_cls)
  - buffered loading from concurrent, out-of-order insert threads
  - recall against brute force, single and batch search, after a pickle round trip
  - reuse of a built index, and rejection of mismatched or stale state
  - rejection of inserts after the build, filters, small dims and missing indexes
"""

from __future__ import annotations

import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("rabitqlib")

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType, NonRetryableInsertError
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.rabitqlib.config import (
    RaBitQLibConfig,
    RaBitQLibHNSWConfig,
)
from vectordb_bench.backend.clients.rabitqlib.rabitqlib_client import RaBitQLib
from vectordb_bench.backend.filter import IntFilter, non_filter

DIM = 128
N = 2000
K = 10


def make_db(path, *, drop_old=True, dim=DIM, **case) -> RaBitQLib:
    case.setdefault("metric_type", MetricType.L2)
    case.setdefault("ef_search", 128)
    case.setdefault("nbits", 7)
    return RaBitQLib(
        dim=dim,
        db_config=RaBitQLibConfig(path=str(path)).to_dict(),
        db_case_config=RaBitQLibHNSWConfig(**case),
        drop_old=drop_old,
    )


def dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    ids = 1000 + 7 * np.arange(N)  # dataset IDs are not row positions
    return vectors, ids


def load(db: RaBitQLib, vectors: np.ndarray, ids: np.ndarray, batch: int = 100) -> None:
    """Insert from four threads in shuffled batch order, as the insert runner may."""
    starts = list(range(0, len(ids), batch))
    random.Random(1).shuffle(starts)
    with db.init():

        def insert(start: int) -> int:
            count, error = db.insert_embeddings(
                vectors[start : start + batch].tolist(), ids[start : start + batch].tolist()
            )
            assert error is None
            return count

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert sum(pool.map(insert, starts)) == len(ids)
    with db.init():
        db.optimize()


def brute_force(vectors: np.ndarray, ids: np.ndarray, queries: np.ndarray, metric: str) -> np.ndarray:
    if metric == "l2":
        order = np.argsort(((queries[:, None, :] - vectors[None]) ** 2).sum(-1), axis=1)
    else:
        order = np.argsort(-(queries @ vectors.T), axis=1)
    return ids[order[:, :K]]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("rabitqlib")
    vectors, ids = dataset()
    db = make_db(path)
    load(db, vectors, ids)
    return path, vectors, ids


class TestConfig:
    @pytest.mark.parametrize(
        ("metric", "expected"),
        [(MetricType.L2, "l2"), (MetricType.IP, "ip"), (MetricType.COSINE, "ip")],
    )
    def test_metric_mapping(self, metric, expected):
        assert RaBitQLibHNSWConfig(metric_type=metric).library_metric() == expected

    def test_rejects_other_metrics(self):
        with pytest.raises(ValueError, match="support L2, IP and COSINE"):
            RaBitQLibHNSWConfig(metric_type=MetricType.HAMMING).index_param()

    def test_search_param_is_not_a_build_param(self):
        config = RaBitQLibHNSWConfig(metric_type=MetricType.L2, ef_search=300)
        assert config.search_param() == {"ef": 300}
        assert "ef" not in config.index_param()

    def test_registration(self):
        assert DB.RaBitQLib.init_cls is RaBitQLib
        assert DB.RaBitQLib.config_cls is RaBitQLibConfig
        assert DB.RaBitQLib.case_config_cls(IndexType.RABITQLIB_HNSW) is RaBitQLibHNSWConfig


class TestBuildAndSearch:
    def test_buffer_is_replaced_by_index(self, built):
        path, _, _ = built
        names = {p.name for p in path.iterdir()}
        assert names == {"rabitqlib_hnsw.index", "rabitqlib_hnsw.id_map.npy", "rabitqlib_hnsw.manifest.json"}

    def test_recall_and_id_mapping(self, built):
        path, vectors, ids = built
        db = pickle.loads(pickle.dumps(make_db(path, drop_old=False)))
        queries = vectors[:50] + 0.05 * np.random.default_rng(2).standard_normal((50, DIM)).astype(np.float32)
        truth = brute_force(vectors, ids, queries, "l2")
        with db.init():
            got = [db.search_embedding(q.tolist(), k=K) for q in queries]
            batch = db.search_embeddings(queries.tolist(), k=K)
        assert got == batch
        recall = np.mean([len(set(g) & set(t)) / K for g, t in zip(got, truth, strict=True)])
        assert recall > 0.9, recall

    def test_unpickling_loads_a_built_index_before_init(self, built, tmp_path):
        # The concurrent runner times from a barrier that precedes init().
        path, _, _ = built
        db = pickle.loads(pickle.dumps(make_db(path, drop_old=False)))
        assert db._index is not None
        assert pickle.loads(pickle.dumps(make_db(tmp_path)))._index is None

    def test_k_above_point_count_drops_empty_slots(self, built):
        path, vectors, ids = built
        db = make_db(path, drop_old=False)
        with db.init():
            result = db.search_embedding(vectors[0].tolist(), k=N + 50)
        # Graph search need not reach every point; the empty slots must not leak out.
        assert N - 50 < len(result) <= N
        assert len(set(result)) == len(result)
        assert set(result) <= set(ids.tolist())

    def test_inner_product(self, tmp_path):
        vectors, ids = dataset(seed=3)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        db = make_db(tmp_path, metric_type=MetricType.COSINE)
        load(db, vectors, ids)
        truth = brute_force(vectors, ids, vectors[:30], "ip")
        with db.init():
            got = db.search_embeddings(vectors[:30].tolist(), k=K)
        recall = np.mean([len(set(g) & set(t)) / K for g, t in zip(got, truth, strict=True)])
        assert recall > 0.9, recall


class TestReuseAndRejection:
    def test_reuse_accepts_other_search_params(self, built):
        path, _, _ = built
        make_db(path, drop_old=False, ef_search=400)

    def test_reuse_rejects_other_build_params(self, built):
        path, _, _ = built
        with pytest.raises(ValueError, match="Rerun with drop_old"):
            make_db(path, drop_old=False, nbits=4)

    def test_insert_after_build_is_rejected(self, built):
        path, vectors, _ = built
        db = make_db(path, drop_old=False)
        with db.init():
            count, error = db.insert_embeddings(vectors[:1].tolist(), [10**9])
        assert count == 0
        assert isinstance(error, NonRetryableInsertError)

    def test_stale_buffer_is_rejected(self, tmp_path):
        vectors, ids = dataset()
        db = make_db(tmp_path)
        with db.init():
            db.insert_embeddings(vectors[:100].tolist(), ids[:100].tolist())
        with pytest.raises(RuntimeError, match="unbuilt load buffer"):
            make_db(tmp_path, drop_old=False)
        make_db(tmp_path, drop_old=True)
        assert not any(tmp_path.iterdir())

    def test_drop_old_keeps_unrelated_files(self, tmp_path):
        # Other data and other index types' files may share the path.
        kept = ["keep.txt", "rabitqlib_ivf.index", "rabitqlib_ivf.manifest.json"]
        for name in kept:
            (tmp_path / name).write_text("not ours")
        make_db(tmp_path)
        assert all((tmp_path / name).exists() for name in kept)

    def test_wrong_vector_dim_is_rejected(self, tmp_path):
        db = make_db(tmp_path)
        with db.init():
            count, error = db.insert_embeddings([[0.0] * (DIM - 1)], [1])
        assert count == 0
        assert isinstance(error, NonRetryableInsertError)

    def test_small_dim_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="dim >= 64"):
            make_db(tmp_path, dim=32)

    def test_filters_are_rejected(self, tmp_path):
        assert RaBitQLib.filter_supported(non_filter)
        assert not RaBitQLib.filter_supported(IntFilter(filter_rate=0.01, int_value=100))
        with pytest.raises(ValueError, match="does not support filtered search"):
            make_db(tmp_path).prepare_filter(IntFilter(filter_rate=0.01, int_value=100))

    def test_search_without_index_raises(self, tmp_path):
        db = make_db(tmp_path)
        with db.init(), pytest.raises(RuntimeError, match="no built rabitqlib_hnsw index"):
            db.search_embedding([0.0] * DIM)
