from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner


class _DefaultDB:
    name = "MockDB"

    def supports_payload_profile(self, payload_profile):
        return True

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        return None

    def search_embedding(self, query, k):
        return [0]


class _InProcessSerialSearchDB(_DefaultDB):
    serial_search_in_process = True


@pytest.mark.parametrize("db_cls", [_DefaultDB, _InProcessSerialSearchDB])
def test_serial_search_runner_run(db_cls):
    runner = SerialSearchRunner(
        db=db_cls(),
        test_data=[[0.1, 0.2]],
        ground_truth=[[0]],
        k=1,
        filters=non_filter,
    )
    recall, ndcg, p99, p95 = runner.run()[0]
    assert recall == 1.0
    assert ndcg == 1.0
    assert p99 >= 0
    assert p95 >= 0


def test_serial_search_runs_in_process_when_client_requests_it():
    runner = SerialSearchRunner(
        db=_InProcessSerialSearchDB(),
        test_data=[[0.1]],
        ground_truth=[[0]],
        k=1,
    )
    with patch.object(SerialSearchRunner, "search", return_value=(0.5, 0.5, 0.01, 0.02)) as mock_search, patch(
        "vectordb_bench.backend.runner.serial_runner.concurrent.futures.ProcessPoolExecutor"
    ) as mock_pool:
        result, _ = runner.run()

    mock_pool.assert_not_called()
    mock_search.assert_called_once()
    assert result == (0.5, 0.5, 0.01, 0.02)


def test_serial_search_uses_subprocess_by_default():
    runner = SerialSearchRunner(
        db=_DefaultDB(),
        test_data=[[0.1]],
        ground_truth=[[0]],
        k=1,
    )
    with patch(
        "vectordb_bench.backend.runner.serial_runner.concurrent.futures.ProcessPoolExecutor"
    ) as mock_pool:
        mock_executor = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value.result.return_value = (0.5, 0.5, 0.01, 0.02)

        result, _ = runner.run()

    mock_pool.assert_called_once_with(max_workers=1)
    assert result == (0.5, 0.5, 0.01, 0.02)
