from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from vectordb_bench import config
from vectordb_bench.backend import task_runner as task_runner_module
from vectordb_bench.backend.cases import CaseLabel, StreamingPerformanceCase
from vectordb_bench.backend.dataset import DataSetIterator, FtsDocumentIterator
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend.runner.serial_runner import SerialInsertRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.backend.workload import WorkloadKind

DEFAULT_INSERT_BATCH_SIZE = config.DEFAULT_INSERT_BATCH_SIZE


class FakeDB:
    name = "FakeDB"
    thread_safe = True

    @contextmanager
    def init(self):
        yield

    def need_normalize_cosine(self):
        return False


def make_case_runner(
    case: Any,
    *,
    batch_size: int = 17,
    load_concurrency: int = 3,
    db: Any | None = None,
) -> CaseRunner:
    task_config = SimpleNamespace(
        insert_batch_size=batch_size,
        load_concurrency=load_concurrency,
        case_config=SimpleNamespace(k=10),
    )
    return CaseRunner.model_construct(ca=case, config=task_config, db=db or FakeDB())


def test_streaming_case_preserves_requested_insert_rate():
    case = StreamingPerformanceCase(insert_rate=550)

    assert case.insert_rate == 550
    assert "550 rows/s" in case.name


def test_direct_callers_use_stable_batch_default():
    dataset = SimpleNamespace(train_files=[])
    concurrent_dataset = SimpleNamespace(data=SimpleNamespace())

    assert DataSetIterator(dataset)._batch_size == DEFAULT_INSERT_BATCH_SIZE
    assert FtsDocumentIterator(SimpleNamespace())._batch_size == DEFAULT_INSERT_BATCH_SIZE
    assert ConcurrentInsertRunner(FakeDB(), concurrent_dataset, normalize=False).batch_size == DEFAULT_INSERT_BATCH_SIZE
    assert SerialInsertRunner(FakeDB(), dataset, normalize=False).batch_size == DEFAULT_INSERT_BATCH_SIZE


def test_serial_insert_runner_groups_rows_by_explicit_batch_size():
    class InsertDB(FakeDB):
        def __init__(self):
            self.metadata_batches = []

        def insert_embeddings(
            self,
            embeddings: list[Any],
            metadata: list[Any],
        ) -> tuple[int, None]:
            self.metadata_batches.append(metadata)
            return len(metadata), None

    db = InsertDB()
    runner = SerialInsertRunner(db, SimpleNamespace(), normalize=False, batch_size=2)

    inserted = runner.endless_insert_data(
        all_embeddings=[[0.1], [0.2], [0.3], [0.4], [0.5]],
        all_metadata=[0, 1, 2, 3, 4],
    )

    assert inserted == 5
    assert db.metadata_batches == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize(
    ("label", "workload_kind"),
    [
        (CaseLabel.Performance, WorkloadKind.VECTOR),
        (CaseLabel.FullTextSearchPerformance, WorkloadKind.FULL_TEXT),
    ],
)
def test_performance_load_propagates_task_batch(
    monkeypatch: pytest.MonkeyPatch,
    label: CaseLabel,
    workload_kind: WorkloadKind,
):
    created: dict[str, Any] = {}

    class FakeConcurrentInsertRunner:
        def __init__(self, *args, **kwargs):
            created.update(kwargs)

        def run(self):
            return 9, 1.25

    case = SimpleNamespace(
        label=label,
        is_multitenant=False,
        dataset=SimpleNamespace(data=SimpleNamespace(metric_type="L2")),
        filters=non_filter,
        load_timeout=30,
        with_scalar_labels=False,
    )
    monkeypatch.setattr(task_runner_module, "ConcurrentInsertRunner", FakeConcurrentInsertRunner)

    result = make_case_runner(case)._load_train_data()

    assert result == (9, 1.25)
    assert created["batch_size"] == 17
    assert created["workload_kind"] == workload_kind


def test_capacity_load_propagates_task_batch(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, Any] = {}

    class FakeSerialInsertRunner:
        def __init__(self, *args, **kwargs):
            created.update(kwargs)

        def run_endlessness(self):
            return 123

    case = SimpleNamespace(
        label=CaseLabel.Load,
        dataset=SimpleNamespace(data=SimpleNamespace(metric_type="L2")),
        filters=non_filter,
        load_timeout=30,
    )
    monkeypatch.setattr(task_runner_module, "SerialInsertRunner", FakeSerialInsertRunner)

    metric = make_case_runner(case)._run_capacity_case()

    assert metric.max_load_count == 123
    assert created["batch_size"] == 17


def test_cloud_insert_propagates_task_batch(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, Any] = {}

    class FakeConcurrentInsertRunner:
        def __init__(self, *args, **kwargs):
            created.update(kwargs)

        def task(self):
            return 3

    class ReadinessDB(FakeDB):
        def poll_insert_readiness(self, expected_count: int) -> dict[str, Any]:
            assert expected_count == 3
            return {"fully_searchable": True, "fully_indexed": True, "additional_parameters": {}}

    case = SimpleNamespace(
        label=CaseLabel.CloudInsert,
        is_multitenant=False,
        dataset=SimpleNamespace(data=SimpleNamespace(metric_type="L2")),
        filters=non_filter,
        duration=60,
        readiness_timeout=None,
        readiness_poll_interval=0,
    )
    monkeypatch.setattr(task_runner_module, "ConcurrentInsertRunner", FakeConcurrentInsertRunner)

    metric = make_case_runner(case, db=ReadinessDB())._run_cloud_insert_case()

    assert metric.inserted_count == 3
    assert created["batch_size"] == 17
    assert created["duration"] == 60


def test_streaming_runner_propagates_task_batch(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, Any] = {}

    class FakeReadWriteRunner:
        def __init__(self, **kwargs):
            created.update(kwargs)

    case = SimpleNamespace(
        label=CaseLabel.Streaming,
        dataset=SimpleNamespace(data=SimpleNamespace(metric_type="L2")),
        insert_rate=34,
        search_stages=[0.5],
        optimize_after_write=False,
        read_dur_after_write=10,
        concurrencies=[1],
    )
    monkeypatch.setattr(task_runner_module, "ReadWriteRunner", FakeReadWriteRunner)

    make_case_runner(case)._init_read_write_runner()

    assert created["insert_rate"] == 34
    assert created["batch_size"] == 17
