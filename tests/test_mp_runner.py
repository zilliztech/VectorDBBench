import logging
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vectordb_bench.backend.runner import mp_runner


def test_worker_records_finish_before_client_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(mp_runner, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    @contextmanager
    def init():
        yield
        clock.now += 30.0

    db = SimpleNamespace(
        init=init,
        supports_payload_profile=lambda _profile: True,
        prepare_filter=lambda _filters: None,
        search_embedding=lambda _query, _k: setattr(clock, "now", 12.0),
    )
    runner = mp_runner.MultiProcessingSearchRunner(db, [[0.0]], duration=1)
    count, total_dur, latencies, finished_at = runner._search_with_finish_time(
        runner.test_data, MagicMock(), MagicMock()
    )
    assert (count, total_dur, latencies, finished_at) == (1, 32.0, [2.0], 12.0)


def test_search_qps_uses_latest_worker_finish_excluding_result_processing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db = SimpleNamespace(supports_payload_profile=lambda _profile: True)
    runner = mp_runner.MultiProcessingSearchRunner(db, [[0.0]], concurrencies=[2], duration=120)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(mp_runner, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    cond = MagicMock()
    cond.notify_all.side_effect = lambda: setattr(clock, "now", 9.0)
    cond.__exit__.side_effect = lambda *_args: setattr(clock, "now", 10.0)
    manager = SimpleNamespace(Queue=lambda: SimpleNamespace(qsize=lambda: 2), Condition=lambda: cond)
    monkeypatch.setattr(mp_runner.mp, "Manager", lambda: nullcontext(manager))

    def worker_result(result: tuple[int, float, list[float], float]) -> SimpleNamespace:
        def collect():
            clock.now = 145.0
            return result

        return SimpleNamespace(result=collect)

    executor = MagicMock()
    executor.submit.side_effect = [
        worker_result((1800, 120.0, [0.01], 130.0)),
        worker_result((1200, 118.0, [0.02], 128.0)),
    ]
    monkeypatch.setattr(mp_runner.concurrent.futures, "ProcessPoolExecutor", lambda **_kwargs: nullcontext(executor))

    latency_summary = runner._latency_summary

    def summarize(latencies: list[float]) -> tuple[float, float, float, float]:
        clock.now = 160.0
        return latency_summary(latencies)

    monkeypatch.setattr(runner, "_latency_summary", summarize)

    with caplog.at_level(logging.INFO, logger=mp_runner.__name__):
        max_qps, concurrencies, qps, *_ = runner.run()

    assert (max_qps, concurrencies, qps) == (25.0, [2], [25.0])
    assert "End search in concurrency 2: dur=120.0s, total_count=3000, qps=25.0" in caplog.messages
    assert "Result processing in concurrency 2: dur=30.0s" in caplog.messages
