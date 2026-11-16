"""Unit tests for the embedded rabitqlib client (HNSW-RaBitQ index).

They build small real indexes with rabitqlib (no server) and cover:
  - metric mapping and registration (init_cls / config_cls / case_config_cls)
  - buffered loading from concurrent, out-of-order insert threads
  - recall against brute force, single and batch search, after a pickle round trip
  - reuse of a built index, and rejection of mismatched or stale state
  - rejection of inserts after the build, filters, small dims and missing indexes
"""

from __future__ import annotations

import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

# CI sets RABITQLIB_REQUIRED where the extra is installed, so a broken install fails instead of skipping.
if os.environ.get("RABITQLIB_REQUIRED"):
    import rabitqlib  # noqa: F401
else:
    pytest.importorskip("rabitqlib")

from vectordb_bench.backend.cases import CaseLabel, CaseType, type2case
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType, NonRetryableInsertError
from vectordb_bench.backend.clients.rabitqlib import cli as rabitqlib_cli
from vectordb_bench.backend.clients.rabitqlib.config import (
    RaBitQLibConfig,
    RaBitQLibHNSWConfig,
    RaBitQLibIndexConfig,
)
from vectordb_bench.backend.clients.rabitqlib.rabitqlib import RaBitQLib
from vectordb_bench.backend.filter import IntFilter, non_filter
from vectordb_bench.cli.vectordbbench import cli

DIM = 128
N = 2000
K = 10


def make_db(path: Path, *, drop_old: bool = True, dim: int = DIM, **case: object) -> RaBitQLib:
    case.setdefault("metric_type", MetricType.L2)
    case.setdefault("ef_search", 128)
    case.setdefault("nbits", 7)
    return RaBitQLib(
        dim=dim,
        db_config=RaBitQLibConfig(path=str(path)).to_dict(),
        db_case_config=RaBitQLibHNSWConfig(**case),
        drop_old=drop_old,
    )


def round_trip(db: RaBitQLib) -> RaBitQLib:
    """Pickle the client into a "new process", as the spawn-based search pool does."""
    return pickle.loads(pickle.dumps(db))  # noqa: S301 - our own object, never untrusted data


Built = tuple[Path, np.ndarray, np.ndarray]


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
def built(tmp_path_factory: pytest.TempPathFactory):
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
    def test_metric_mapping(self, metric: MetricType, expected: str):
        assert RaBitQLibHNSWConfig(metric_type=metric).library_metric() == expected

    def test_rejects_other_metrics(self):
        with pytest.raises(ValueError, match="support L2, IP and COSINE"):
            RaBitQLibHNSWConfig(metric_type=MetricType.HAMMING).index_param()

    def test_search_param_is_not_a_build_param(self):
        config = RaBitQLibHNSWConfig(metric_type=MetricType.L2, ef_search=300)
        assert config.search_param() == {"ef": 300}
        assert "ef" not in config.index_param()

    def test_base_config_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            RaBitQLibIndexConfig(index=IndexType.RABITQLIB_HNSW)

    def test_search_kwargs_come_from_search_param(self):
        config = RaBitQLibHNSWConfig(metric_type=MetricType.L2, ef_search=300)
        assert config.search_kwargs(k=10) == {"ef": 300}
        assert config.search_kwargs(k=500) == {"ef": 500}

    def test_registration(self):
        assert DB.RaBitQLib.init_cls is RaBitQLib
        assert DB.RaBitQLib.config_cls is RaBitQLibConfig
        assert DB.RaBitQLib.case_config_cls(IndexType.RABITQLIB_HNSW) is RaBitQLibHNSWConfig


class TestBuildAndSearch:
    def test_buffer_is_replaced_by_index(self, built: Built):
        path, _, _ = built
        names = {p.name for p in path.iterdir()}
        assert names == {"rabitqlib_hnsw.index", "rabitqlib_hnsw.id_map.npy", "rabitqlib_hnsw.manifest.json"}

    def test_recall_and_id_mapping(self, built: Built):
        path, vectors, ids = built
        db = round_trip(make_db(path, drop_old=False))
        queries = vectors[:50] + 0.05 * np.random.default_rng(2).standard_normal((50, DIM)).astype(np.float32)
        truth = brute_force(vectors, ids, queries, "l2")
        with db.init():
            got = [db.search_embedding(q.tolist(), k=K) for q in queries]
            batch = db.search_embeddings(queries.tolist(), k=K)
        assert got == batch
        recall = np.mean([len(set(g) & set(t)) / K for g, t in zip(got, truth, strict=True)])
        assert recall > 0.9, recall

    def test_unpickling_loads_a_built_index_before_init(self, built: Built, tmp_path: Path):
        # The concurrent runner times from a barrier that precedes init().
        path, _, _ = built
        db = round_trip(make_db(path, drop_old=False))
        assert db._index is not None
        assert round_trip(make_db(tmp_path))._index is None

    def test_k_above_point_count_drops_empty_slots(self, built: Built):
        path, vectors, ids = built
        db = make_db(path, drop_old=False)
        with db.init():
            result = db.search_embedding(vectors[0].tolist(), k=N + 50)
        # Graph search need not reach every point; the empty slots must not leak out.
        assert N - 50 < len(result) <= N
        assert len(set(result)) == len(result)
        assert set(result) <= set(ids.tolist())

    def test_inner_product(self, tmp_path: Path):
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
    def test_reuse_accepts_other_search_params(self, built: Built):
        path, _, _ = built
        make_db(path, drop_old=False, ef_search=400)

    def test_reuse_rejects_other_build_params(self, built: Built):
        path, _, _ = built
        with pytest.raises(ValueError, match="Rerun with drop_old"):
            make_db(path, drop_old=False, nbits=4)

    def test_reuse_rejects_other_dim(self, built: Built):
        path, _, _ = built
        with pytest.raises(ValueError, match="Rerun with drop_old"):
            make_db(path, drop_old=False, dim=DIM * 2)

    @pytest.mark.parametrize(
        ("built_as", "reused_as"),
        [(MetricType.COSINE, MetricType.IP), (MetricType.IP, MetricType.COSINE)],
    )
    def test_reuse_rejects_cosine_ip_swap(self, tmp_path: Path, built_as: MetricType, reused_as: MetricType):
        # Both build an "ip" index with equal build params; only COSINE stores normalized vectors.
        assert (
            RaBitQLibHNSWConfig(metric_type=built_as).index_param()
            == RaBitQLibHNSWConfig(metric_type=reused_as).index_param()
        )
        vectors, ids = dataset()
        load(make_db(tmp_path, metric_type=built_as), vectors, ids)
        with pytest.raises(ValueError, match=f"'metric_type': '{reused_as.value}'"):
            make_db(tmp_path, drop_old=False, metric_type=reused_as)
        make_db(tmp_path, drop_old=False, metric_type=built_as, ef_search=300)

    def test_insert_after_build_is_rejected(self, built: Built):
        path, vectors, _ = built
        db = make_db(path, drop_old=False)
        with db.init():
            count, error = db.insert_embeddings(vectors[:1].tolist(), [10**9])
        assert count == 0
        assert isinstance(error, NonRetryableInsertError)

    def test_stale_buffer_is_rejected(self, tmp_path: Path):
        vectors, ids = dataset()
        db = make_db(tmp_path)
        with db.init():
            db.insert_embeddings(vectors[:100].tolist(), ids[:100].tolist())
        with pytest.raises(RuntimeError, match="unbuilt load buffer"):
            make_db(tmp_path, drop_old=False)
        make_db(tmp_path, drop_old=True)
        assert not any(tmp_path.iterdir())

    def test_drop_old_keeps_unrelated_files(self, tmp_path: Path):
        # Other data and other index types' files may share the path.
        kept = [
            "keep.txt",
            "rabitqlib_ivf.index",
            "rabitqlib_ivf.manifest.json",
            "rabitqlib_ivf.buffer.vectors.f32",
            "rabitqlib_ivf.buffer.ids.i64",
        ]
        for name in kept:
            (tmp_path / name).write_text("not ours")
        make_db(tmp_path)
        assert all((tmp_path / name).exists() for name in kept)

    def test_another_index_types_buffer_is_not_stale(self, tmp_path: Path):
        # An unbuilt load of another index type at the same path is not ours to refuse.
        (tmp_path / "rabitqlib_ivf.buffer.ids.i64").write_bytes(b"")
        make_db(tmp_path, drop_old=False)

    def test_manifest_from_another_version_is_rejected(self, built: Built):
        path, _, _ = built
        manifest_file = path / "rabitqlib_hnsw.manifest.json"
        original = manifest_file.read_text()
        try:
            manifest_file.write_text(json.dumps({**json.loads(original), "version": 0}))
            with pytest.raises(ValueError, match="has version 0"):
                make_db(path, drop_old=False)
        finally:
            manifest_file.write_text(original)

    def test_manifest_records_build_threads_as_given(self, built: Built):
        path, _, _ = built
        assert json.loads((path / "rabitqlib_hnsw.manifest.json").read_text())["build_threads"] == "auto"

    def test_optimize_without_loading_explains(self, tmp_path: Path):
        db = make_db(tmp_path)
        with db.init(), pytest.raises(RuntimeError, match="no load buffer"):
            db.optimize()

    def test_wrong_vector_dim_is_rejected(self, tmp_path: Path):
        db = make_db(tmp_path)
        with db.init():
            count, error = db.insert_embeddings([[0.0] * (DIM - 1)], [1])
        assert count == 0
        assert isinstance(error, NonRetryableInsertError)

    def test_small_dim_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="dim >= 64"):
            make_db(tmp_path, dim=32)

    def test_filters_are_rejected(self, tmp_path: Path):
        assert RaBitQLib.filter_supported(non_filter)
        assert not RaBitQLib.filter_supported(IntFilter(filter_rate=0.01, int_value=100))
        with pytest.raises(ValueError, match="does not support filtered search"):
            make_db(tmp_path).prepare_filter(IntFilter(filter_rate=0.01, int_value=100))

    @pytest.mark.parametrize(
        "case_type",
        [
            "CapacityDim128",
            "CapacityDim960",
            "StreamingPerformanceCase",
            "StreamingCustomDataset",
            "CloudInsertCase",
            "CloudColdLatencyCase",
            "FTSBm25Performance",
        ],
    )
    def test_cli_rejects_non_performance_cases_before_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_type: str
    ):
        # Capacity would only grow the buffer; streaming would search before the build.
        called = []
        monkeypatch.setattr(rabitqlib_cli, "run", lambda **kwargs: called.append(kwargs))
        result = CliRunner().invoke(cli, ["rabitqlibhnsw", "--path", str(tmp_path), "--case-type", case_type])
        assert result.exit_code == 2, result.output
        assert "performance cases only" in result.output
        assert not called
        assert not any(tmp_path.iterdir())

    @pytest.mark.parametrize(
        "case_type",
        [
            ct.name
            for ct in CaseType
            if ct in type2case and type2case[ct].model_fields["label"].default == CaseLabel.Performance
        ],
    )
    def test_cli_accepts_performance_cases(self, case_type: str):
        rabitqlib_cli.require_performance_case(case_type)

    def test_search_without_index_raises(self, tmp_path: Path):
        db = make_db(tmp_path)
        with db.init(), pytest.raises(RuntimeError, match="no built rabitqlib_hnsw index"):
            db.search_embedding([0.0] * DIM)
