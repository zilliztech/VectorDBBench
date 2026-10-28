"""Tests for the KiviDB client.

The unit tests need nothing. The `integration` tests need a KiviDB server
(>= 1.0.5) and are skipped when none is reachable:

    docker run -d -p 6380:6380 ghcr.io/kividbio/kividb:v1.1.2-full
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
from pydantic import SecretStr

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB, IndexType, MetricType
from vectordb_bench.backend.clients.kividb.config import KiviDBConfig, KiviDBFLATConfig, KiviDBHNSWConfig
from vectordb_bench.backend.clients.kividb.kividb import escape_tag_value
from vectordb_bench.backend.filter import FilterOp, IntFilter, LabelFilter, non_filter
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import InputType, get_case_config_inputs
from vectordb_bench.models import CaseConfig

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


def test_autoindex_falls_back_to_hnsw():
    assert DB.KiviDB.case_config_cls(IndexType.AUTOINDEX) is KiviDBHNSWConfig


@pytest.mark.parametrize(
    ("index", "expected"),
    [(None, KiviDBHNSWConfig), (IndexType.HNSW.value, KiviDBHNSWConfig), (IndexType.Flat.value, KiviDBFLATConfig)],
    ids=["default", "hnsw", "flat"],
)
def test_frontend_generates_tasks(index: str | None, expected: type):
    """Fill the KiviDB inputs with their defaults, as the run-test page does, then generate the task."""
    case = CaseConfig(case_id=CaseType.Performance768D1M)
    ui: dict = {}
    for inp in get_case_config_inputs(DB.KiviDB, case.case.label):
        if inp.isDisplayed(ui):
            if inp.inputType == InputType.Option:
                ui[inp.label] = index or inp.inputConfig["options"][0]
            else:
                ui[inp.label] = inp.inputConfig["value"]
    db_config = KiviDBConfig(db_label="test", host=SecretStr("localhost"))
    (task,) = generate_tasks([DB.KiviDB], {DB.KiviDB: db_config}, [case], {DB.KiviDB: {case: ui}}, batch_size=100)
    assert type(task.db_case_config) is expected
    if expected is KiviDBHNSWConfig:
        assert task.db_case_config.index_param()["m"] == 16
        assert task.db_case_config.index_param()["ef_construction"] == 200
        assert task.db_case_config.search_param() == {"ef_runtime": 10}


class _FakeConn:
    closed = False

    def close(self):
        self.closed = True


def test_init_closes_connection_when_body_raises(monkeypatch: pytest.MonkeyPatch):
    db = DB.KiviDB.init_cls.__new__(DB.KiviDB.init_cls)
    db.case_config = KiviDBHNSWConfig()
    conn = _FakeConn()
    monkeypatch.setattr(db, "_connect", lambda: conn)

    def insert_fails():
        with db.init():
            assert db.conn is conn
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        insert_fails()
    assert conn.closed
    assert db.conn is None


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
