from types import SimpleNamespace

import pytest

from vectordb_bench import config
from vectordb_bench.backend.runner import read_write_runner as read_write_runner_module
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.runner.read_write_runner import ReadWriteRunner

DEFAULT_INSERT_BATCH_SIZE = config.DEFAULT_INSERT_BATCH_SIZE


class FakeDB:
    name = "FakeDB"


def test_rate_runner_uses_explicit_batch_size():
    runner = RatedMultiThreadingInsertRunner(
        rate=30,
        db=FakeDB(),
        dataset_iter=iter(()),
        batch_size=5,
    )

    assert runner.insert_rate == 30
    assert runner.batch_size == 5
    assert runner.batch_rate == 6


def test_rate_runner_direct_caller_uses_stable_batch_default():
    runner = RatedMultiThreadingInsertRunner(
        rate=DEFAULT_INSERT_BATCH_SIZE,
        db=FakeDB(),
        dataset_iter=iter(()),
    )

    assert runner.batch_size == DEFAULT_INSERT_BATCH_SIZE
    assert runner.batch_rate == 1


@pytest.mark.parametrize(
    ("rate", "batch_size", "message"),
    [
        (0, 10, "insert rate must be greater than 0"),
        (-10, 10, "insert rate must be greater than 0"),
        (10, 0, "insert batch size must be greater than 0"),
        (10, -1, "insert batch size must be greater than 0"),
        (10, 4, "insert rate 10 must be divisible by insert batch size 4"),
    ],
)
def test_rate_runner_rejects_invalid_rate_batch_combinations(rate, batch_size, message):
    with pytest.raises(ValueError, match=message):
        RatedMultiThreadingInsertRunner(
            rate=rate,
            db=FakeDB(),
            dataset_iter=iter(()),
            batch_size=batch_size,
        )


def test_read_write_runner_requests_task_batch_from_dataset(monkeypatch):
    requested_batch_sizes = []

    class Dataset:
        data = SimpleNamespace(size=100)
        test_data = []
        gt_data = []

        def iter_batches(self, batch_size):
            requested_batch_sizes.append(batch_size)
            return iter(())

    class FakeSerialSearchRunner:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(MultiProcessingSearchRunner, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(read_write_runner_module, "SerialSearchRunner", FakeSerialSearchRunner)

    runner = ReadWriteRunner(
        db=FakeDB(),
        dataset=Dataset(),
        insert_rate=30,
        batch_size=5,
    )

    assert requested_batch_sizes == [5]
    assert runner.batch_size == 5
    assert runner.batch_rate == 6


def test_streaming_search_returns_per_stage_recall_stats():
    stats = {"recall_min": 0.0, "recall_p1": 0.1, "recall_p5": 0.2, "recall_p10": 0.3}
    runner = ReadWriteRunner.__new__(ReadWriteRunner)
    runner.serial_search_runner = SimpleNamespace(run=lambda: ((0.5, 0.5, 0.2, 0.1, 0.05, {}, stats), 1.0))
    runner.read_dur_after_write = 1
    runner.run_by_dur = lambda duration: (10, 0, [1], [10], [0.2], [0.1], [0.05])

    assert runner.run_search(100)[0][-1] == stats

    runner.data_volume = 200
    runner.insert_rate = 100
    runner.search_stages = [0.5]
    runner.get_each_conc_search_dur = lambda *args, **kwargs: 0
    queue = SimpleNamespace(get=lambda **kwargs: False, empty=lambda: True)
    assert runner.run_search_by_sig(queue)[0][-1] == stats


def test_streaming_metrics_preserve_recall_stats_and_missing_stage(monkeypatch):
    stats = {"recall_min": 0.0, "recall_p1": 0.1, "recall_p5": 0.2, "recall_p10": 0.3}
    stage = (50, 0, 0, 0.5, 0, 0, 0, 0, [], [], [], [], [], stats)
    missing = (100, 0, 0, 0, 0, 0, 0, 0, [], [], [], [], [], {})

    class Manager:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def Queue(self):
            return object()

    class Executor:
        def __init__(self, **kwargs):
            self.results = iter([(200, 2.0), [stage], [missing]])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args):
            value = next(self.results)
            return SimpleNamespace(result=lambda: value)

    monkeypatch.setattr(read_write_runner_module.mp, "Manager", Manager)
    monkeypatch.setattr(read_write_runner_module.concurrent.futures, "ProcessPoolExecutor", Executor)
    runner = ReadWriteRunner.__new__(ReadWriteRunner)
    runner.data_volume = 200
    runner.insert_rate = 100
    runner.optimize_after_write = False

    metrics = runner.run_read_write()

    assert metrics.st_recall_min_list == [0.0, None]
    assert metrics.st_recall_p1_list == [0.1, None]
    assert metrics.st_recall_p5_list == [0.2, None]
    assert metrics.st_recall_p10_list == [0.3, None]
