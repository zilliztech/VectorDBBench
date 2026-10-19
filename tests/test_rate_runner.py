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
