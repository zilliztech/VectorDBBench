"""Tests for the KiviDB client.

The unit tests need nothing. The `integration` tests need a KiviDB server
(>= 1.0.5) and are skipped when none is reachable:

    docker run -d -p 6380:6380 quay.io/kividbio/kividb:v1.0.5-full
    KIVIDB_HOST=localhost KIVIDB_PORT=6380 pytest tests/test_kividb.py -v

Recall is checked against brute-force ground truth over the exact vectors
inserted, so a filter that is dropped or mis-built fails here rather than
producing a plausible-looking number.
"""

import os
from collections.abc import Callable

import numpy as np
import pytest
import redis

from vectordb_bench.backend.clients import DB, IndexType, MetricType
from vectordb_bench.backend.clients.kividb.config import KiviDBFLATConfig, KiviDBHNSWConfig
from vectordb_bench.backend.clients.kividb.kividb import escape_tag_value
from vectordb_bench.backend.filter import FilterOp, IntFilter, LabelFilter, non_filter

HOST = os.environ.get("KIVIDB_HOST", "localhost")
PORT = int(os.environ.get("KIVIDB_PORT", "6380"))
DIM, COUNT, NQ, K = 32, 2000, 20, 10
LABELS = ["label_1p", "label_5p", "label_50p"]


# ---------------------------------------------------------------- unit tests
def test_registered_with_the_benchmark():
    assert DB.KiviDB.value == "KiviDB"
    assert DB.KiviDB.init_cls.name == "KiviDB"
    assert DB.KiviDB.config_cls.__name__ == "KiviDBConfig"
    assert DB.KiviDB.case_config_cls(IndexType.HNSW) is KiviDBHNSWConfig
    assert DB.KiviDB.case_config_cls(IndexType.Flat) is KiviDBFLATConfig


def test_supported_filters():
    cls = DB.KiviDB.init_cls
    assert set(cls.supported_filter_types) == {FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual}


def test_metric_mapping():
    assert KiviDBHNSWConfig(metric_type=MetricType.L2).index_param()["metric"] == "L2"
    assert KiviDBHNSWConfig(metric_type=MetricType.IP).index_param()["metric"] == "IP"
    assert KiviDBHNSWConfig(metric_type=MetricType.COSINE).index_param()["metric"] == "COSINE"


def test_tag_values_are_escaped():
    assert escape_tag_value("label_5p") == "label_5p"
    assert escape_tag_value("a-b c|d") == "a\\-b\\ c\\|d"


# ---------------------------------------------------------- integration tests
def _server_available() -> bool:
    try:
        return "kividb_version" in redis.Redis(host=HOST, port=PORT, socket_timeout=2).info("server")
    except Exception:
        return False


live = pytest.mark.skipif(not _server_available(), reason=f"no KiviDB server at {HOST}:{PORT}")


def _data():
    rng = np.random.default_rng(7)
    vectors = rng.random((COUNT, DIM), dtype=np.float32)
    queries = rng.random((NQ, DIM), dtype=np.float32)
    labels = [LABELS[i % len(LABELS)] for i in range(COUNT)]
    return vectors, queries, labels


def _brute_force(vectors: np.ndarray, query: np.ndarray, candidates: list[int], k: int) -> list[int]:
    v = vectors[candidates]
    sims = (v @ query) / (np.linalg.norm(v, axis=1) * np.linalg.norm(query))
    return [candidates[i] for i in np.argsort(-sims)[:k]]


def _client(with_scalar_labels: bool = False):
    cfg = KiviDBHNSWConfig(metric_type=MetricType.COSINE, M=16, ef_construction=200, ef_runtime=200)
    return DB.KiviDB.init_cls(
        dim=DIM,
        db_config={"host": HOST, "port": PORT, "password": None, "ssl": False},
        db_case_config=cfg,
        collection_name="vdbbench_kividb_test",
        drop_old=True,
        with_scalar_labels=with_scalar_labels,
    )


def _recall(
    db: object,
    vectors: np.ndarray,
    queries: np.ndarray,
    candidates: list[int],
    check: Callable[[int], bool] | None = None,
) -> float:
    hits = 0
    for q in queries:
        got = db.search_embedding(q.tolist(), k=K)
        if check:
            assert all(check(i) for i in got), f"result outside the filter: {got}"
        hits += len(set(got) & set(_brute_force(vectors, q, candidates, K)))
    return hits / (len(queries) * K)


@pytest.mark.integration
@live
@pytest.mark.parametrize(
    ("filters", "candidates_of", "check_of"),
    [
        (non_filter, lambda _labels: list(range(COUNT)), None),
        (
            IntFilter(int_value=1500, filter_rate=0.75),
            lambda _labels: list(range(1500, COUNT)),
            lambda _labels: lambda i: i >= 1500,
        ),
        (
            LabelFilter(label_percentage=0.05),
            lambda labels: [i for i in range(COUNT) if labels[i] == "label_5p"],
            lambda labels: lambda i: labels[i] == "label_5p",
        ),
    ],
    ids=["no-filter", "int-ge", "label-eq"],
)
def test_insert_search_recall(
    filters: object,
    candidates_of: Callable[[list[str]], list[int]],
    check_of: Callable[[list[str]], Callable[[int], bool]] | None,
):
    vectors, queries, labels = _data()
    db = _client(with_scalar_labels=filters.type == FilterOp.StrEqual)
    with db.init():
        n, err = db.insert_embeddings(vectors.tolist(), list(range(COUNT)), labels_data=labels)
        assert (n, err) == (COUNT, None)
    db.optimize(data_size=COUNT)
    with db.init():
        db.prepare_filter(filters)
        check = check_of(labels) if check_of else None
        recall = _recall(db, vectors, queries, candidates_of(labels), check)
    assert recall >= 0.95, f"recall {recall:.3f} < 0.95 for {filters.type}"


@pytest.mark.integration
@live
def test_drop_old_removes_previous_corpus():
    vectors, _, labels = _data()
    db = _client()
    with db.init():
        db.insert_embeddings(vectors.tolist(), list(range(COUNT)), labels_data=labels)
    _client()  # drop_old=True again
    conn = redis.Redis(host=HOST, port=PORT)
    assert not list(conn.scan_iter(match="vdbbench_kividb_test:*", count=1000))
