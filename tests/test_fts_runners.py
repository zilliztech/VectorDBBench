from types import SimpleNamespace
import pickle
from queue import Empty
import time

import pytest

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialFtsInsertRunner, SerialSearchRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.backend.workload import WorkloadKind
from vectordb_bench.models import ConcurrencySlotTimeoutError, LoadTimeoutError, PerformanceTimeoutError, TaskStage


class FakeDB:
    name = "FakeDB"

    def __init__(self, has_text_field=True):
        self.calls = []
        self._has_text_field = has_text_field

    def search_embedding(self, query, k=100):
        self.calls.append(("vector", query, k))
        return ["1"]

    def search_documents(self, query, k=100, **kwargs):
        self.calls.append(("fts", query, k, kwargs))
        return ["doc-1"]

    def need_normalize_cosine(self):
        return False

    def supports_payload_profile(self, payload_profile):
        return True

    def has_text_field(self):
        return self._has_text_field

    def supports_document_payload_profile(self, payload_profile):
        if payload_profile == PayloadProfile.IDS_ONLY:
            return True
        if payload_profile == PayloadProfile.TEXT:
            return self.has_text_field()
        return False


class FakeInsertRunner:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run(self):
        return None


def make_case_runner(label):
    return CaseRunner.construct(ca=SimpleNamespace(label=label))


def test_serial_runner_uses_explicit_vector_workload():
    db = FakeDB()
    runner = SerialSearchRunner(db=db, test_data=[[0.1]], ground_truth=[["1"]], k=3, workload_kind=WorkloadKind.VECTOR)

    assert runner._get_db_search_res([0.1]) == ["1"]
    assert db.calls == [("vector", [0.1], 3)]


def test_serial_runner_uses_explicit_fts_workload():
    db = FakeDB()
    runner = SerialSearchRunner(
        db=db,
        test_data=["alpha"],
        ground_truth=[["doc-1"]],
        k=5,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    assert runner._get_db_search_res("alpha") == ["doc-1"]
    assert db.calls == [("fts", "alpha", 5, {})]
    assert runner._use_fts_metrics is True


def test_serial_runner_passes_text_payload_for_fts_workload():
    db = FakeDB()
    runner = SerialSearchRunner(
        db=db,
        test_data=["alpha"],
        ground_truth=[["doc-1"]],
        k=5,
        payload_profile=PayloadProfile.TEXT,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    assert runner._get_db_search_res("alpha") == ["doc-1"]
    assert db.calls == [("fts", "alpha", 5, {"payload_profile": PayloadProfile.TEXT})]


def test_serial_runner_rejects_text_payload_when_fts_backend_has_no_text_field():
    with pytest.raises(NotImplementedError, match="document payload_profile=text"):
        SerialSearchRunner(
            db=FakeDB(has_text_field=False),
            test_data=["alpha"],
            ground_truth=[["doc-1"]],
            payload_profile=PayloadProfile.TEXT,
            workload_kind=WorkloadKind.FULL_TEXT_BM25,
        )


def test_mp_runner_uses_explicit_fts_workload():
    db = FakeDB()
    runner = MultiProcessingSearchRunner(
        db=db,
        test_data=["alpha"],
        k=5,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    assert runner._search_func("alpha", 5) == ["doc-1"]


def test_mp_runner_passes_text_payload_for_fts_workload():
    db = FakeDB()
    runner = MultiProcessingSearchRunner(
        db=db,
        test_data=["alpha"],
        k=5,
        payload_profile=PayloadProfile.TEXT,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    assert runner._search_once("alpha") == ["doc-1"]
    assert db.calls == [("fts", "alpha", 5, {"payload_profile": PayloadProfile.TEXT})]


def test_mp_runner_rejects_text_payload_when_fts_backend_has_no_text_field():
    with pytest.raises(NotImplementedError, match="document payload_profile=text"):
        MultiProcessingSearchRunner(
            db=FakeDB(has_text_field=False),
            test_data=["alpha"],
            payload_profile=PayloadProfile.TEXT,
            workload_kind=WorkloadKind.FULL_TEXT_BM25,
        )


def test_fts_runners_reject_vector_payload_for_document_search():
    with pytest.raises(NotImplementedError, match="document payload_profile=vector"):
        SerialSearchRunner(
            db=FakeDB(),
            test_data=["alpha"],
            ground_truth=[["doc-1"]],
            payload_profile=PayloadProfile.VECTOR,
            workload_kind=WorkloadKind.FULL_TEXT_BM25,
        )


def test_fts_and_vector_perf_paths_use_same_orchestration_methods():
    assert hasattr(CaseRunner, "_run_perf_case")
    assert not hasattr(CaseRunner, "_run_fts_perf_case")


def test_fts_load_data_dispatches_to_fts_loader():
    runner = make_case_runner(CaseLabel.FullTextSearchPerformance)
    calls = []

    object.__setattr__(runner, "_load_fts_data", lambda: calls.append("fts"))
    object.__setattr__(runner, "_load_train_data", lambda: calls.append("vector"))

    result, _ = runner._load_data()

    assert result is None
    assert calls == ["fts"]


def test_vector_load_data_dispatches_to_vector_loader():
    runner = make_case_runner(CaseLabel.Performance)
    calls = []

    object.__setattr__(runner, "_load_fts_data", lambda: calls.append("fts"))
    object.__setattr__(runner, "_load_train_data", lambda: calls.append("vector"))

    result, _ = runner._load_data()

    assert result is None
    assert calls == ["vector"]


def test_fts_init_search_runners_dispatches_to_fts_initializer():
    runner = make_case_runner(CaseLabel.FullTextSearchPerformance)
    calls = []

    object.__setattr__(runner, "_init_fts_search_runner", lambda: calls.append("fts"))
    object.__setattr__(runner, "_init_search_runner", lambda: calls.append("vector"))

    runner._init_search_runners()

    assert calls == ["fts"]


def test_vector_init_search_runners_dispatches_to_vector_initializer():
    runner = make_case_runner(CaseLabel.Performance)
    calls = []

    object.__setattr__(runner, "_init_fts_search_runner", lambda: calls.append("fts"))
    object.__setattr__(runner, "_init_search_runner", lambda: calls.append("vector"))

    runner._init_search_runners()

    assert calls == ["vector"]


def test_fts_init_search_runner_passes_payload_profile(monkeypatch):
    from vectordb_bench.backend import task_runner

    captured = {}

    class CaptureRunner:
        def __init__(self, **kwargs):
            captured.setdefault(kwargs["workload_kind"], []).append(kwargs)

    monkeypatch.setattr(task_runner, "SerialSearchRunner", CaptureRunner)
    monkeypatch.setattr(task_runner, "MultiProcessingSearchRunner", CaptureRunner)

    runner = CaseRunner.construct(
        db=FakeDB(),
        ca=SimpleNamespace(
            dataset=SimpleNamespace(
                queries_data=[SimpleNamespace(query_id="q1", text="alpha")],
                qrels_data={"q1": ["doc-1"]},
            ),
            filters="filters",
            payload_profile=PayloadProfile.TEXT,
        ),
        config=SimpleNamespace(
            stages=[TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
            case_config=SimpleNamespace(
                k=7,
                concurrency_search_config=SimpleNamespace(
                    num_concurrency=[1],
                    concurrency_duration=2,
                    concurrency_timeout=3,
                ),
            ),
        ),
    )

    runner._init_fts_search_runner()

    serial_kwargs, concurrent_kwargs = captured[WorkloadKind.FULL_TEXT_BM25]
    assert serial_kwargs["payload_profile"] == PayloadProfile.TEXT
    assert concurrent_kwargs["payload_profile"] == PayloadProfile.TEXT


def test_fts_run_routes_to_shared_perf_case():
    runner = make_case_runner(CaseLabel.FullTextSearchPerformance)
    calls = []

    object.__setattr__(runner, "_pre_run", lambda drop_old=True: calls.append(("pre", drop_old)))
    object.__setattr__(runner, "_run_perf_case", lambda drop_old=True: calls.append(("perf", drop_old)) or "metric")

    assert runner.run(drop_old=False) == "metric"
    assert calls == [("pre", False), ("perf", False)]


def test_mp_runner_rebuilds_search_function_after_pickle():
    runner = MultiProcessingSearchRunner(
        db=FakeDB(),
        test_data=["query"],
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    restored = pickle.loads(pickle.dumps(runner))

    assert restored._search_once("alpha") == ["doc-1"]


def test_mp_runner_waits_for_ready_signals_without_qsize():
    class ReadyQueue:
        def __init__(self):
            self.values = [1, 1]

        def get(self, timeout=None):
            if not self.values:
                raise Empty
            return self.values.pop()

    runner = MultiProcessingSearchRunner(
        db=FakeDB(),
        test_data=["query"],
        concurrency_timeout=1,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    runner._wait_for_queue_fill(ReadyQueue(), size=2)


def test_mp_runner_ready_wait_times_out_when_workers_do_not_signal():
    class EmptyQueue:
        def get(self, timeout=None):
            raise Empty

    runner = MultiProcessingSearchRunner(
        db=FakeDB(),
        test_data=["query"],
        concurrency_timeout=0.01,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    with pytest.raises(ConcurrencySlotTimeoutError):
        runner._wait_for_queue_fill(EmptyQueue(), size=1)


def test_vector_load_train_data_is_not_individually_timed(monkeypatch):
    from vectordb_bench.backend import task_runner

    monkeypatch.setattr(task_runner, "ConcurrentInsertRunner", FakeInsertRunner)
    runner = CaseRunner.construct(
        db=FakeDB(),
        config=SimpleNamespace(load_concurrency=0),
        ca=SimpleNamespace(
            label=CaseLabel.Performance,
            dataset=SimpleNamespace(data=SimpleNamespace(metric_type=None)),
            filters="filters",
            is_multitenant=False,
            load_timeout=1,
            with_scalar_labels=False,
        ),
    )

    assert runner._load_train_data() is None


def test_fts_load_fts_data_is_not_individually_timed(monkeypatch):
    from vectordb_bench.backend import task_runner

    monkeypatch.setattr(task_runner, "SerialFtsInsertRunner", FakeInsertRunner)
    runner = CaseRunner.construct(
        db=FakeDB(),
        ca=SimpleNamespace(dataset="dataset", load_timeout=1),
    )

    assert runner._load_fts_data() is None


def test_fts_load_data_gets_single_timing_boundary(monkeypatch):
    from vectordb_bench.backend import task_runner

    monkeypatch.setattr(task_runner, "SerialFtsInsertRunner", FakeInsertRunner)
    runner = CaseRunner.construct(
        db=FakeDB(),
        ca=SimpleNamespace(label=CaseLabel.FullTextSearchPerformance, dataset="dataset", load_timeout=1),
    )

    result, _ = runner._load_data()

    assert result is None


def test_fts_insert_runner_enforces_load_timeout(monkeypatch):
    runner = SerialFtsInsertRunner(db=FakeDB(), dataset=[], timeout=0.01)

    def slow_insert():
        time.sleep(1)
        return 0, 1.0

    monkeypatch.setattr(runner, "_insert_all_batches", slow_insert)

    with pytest.raises(LoadTimeoutError):
        runner.run()


def test_fts_optimize_enforces_optimize_timeout(monkeypatch):
    runner = make_case_runner(CaseLabel.FullTextSearchPerformance)
    object.__setattr__(runner, "ca", SimpleNamespace(label=CaseLabel.FullTextSearchPerformance, optimize_timeout=0.01))

    def slow_optimize(self):
        time.sleep(1)
        return None, 1.0

    monkeypatch.setattr(CaseRunner, "_optimize_task", slow_optimize)

    with pytest.raises(PerformanceTimeoutError):
        runner._optimize()
