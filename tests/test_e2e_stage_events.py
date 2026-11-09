import json
import os
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_milvus import TestMilvusOptimize as OptimizeFixture

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.stage_events import PREFIX, stage
from vectordb_bench.backend.task_runner import CaseRunner


def records(output: str):
    return [json.loads(line[len(PREFIX) :]) for line in output.splitlines() if line.startswith(PREFIX)]


def test_stage_events_are_opt_in_and_report_failure(monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]):
    monkeypatch.delenv("E2E_STAGE_EVENTS", raising=False)
    with stage("insert"):
        pass
    assert not capfd.readouterr().out
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_RUN_ID", "run")
    monkeypatch.setenv("E2E_CASE_ID", "case")
    monkeypatch.setenv("E2E_ATTEMPT", "2")
    with pytest.raises(ValueError, match="failure"), stage("search_concurrent", concurrency=5):
        raise ValueError("failure")
    start, end = records(capfd.readouterr().out)
    assert (start["state"], end["state"]) == ("start", "failed")
    assert start["span_id"] == end["span_id"]
    assert (start["run_id"], start["case_id"], start["attempt"], start["concurrency"]) == ("run", "case", 2, 5)
    assert end["monotonic_ns"] >= start["monotonic_ns"]


def test_stage_events_cross_subprocess_stdout():
    env = {**os.environ, "E2E_STAGE_EVENTS": "1", "E2E_RUN_ID": "child"}
    code = "from vectordb_bench.backend.stage_events import stage\nwith stage('compaction'): pass"
    child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    events = records(child.stdout)
    assert [event["state"] for event in events] == ["start", "end"]
    assert all(event["run_id"] == "child" for event in events)


def test_compaction_records_before_after_without_changing_timing(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.list_persistent_segments.side_effect = [[1, 2, 3], [4]]
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.perf_counter", side_effect=[10, 13]):
        db._optimize()
    events = records(capfd.readouterr().out)
    assert [event["state"] for event in events] == ["start", "end", "segments"]
    observation = db.optimization_metrics["compaction_segments"][0]
    assert observation["before"]["count"] == 3
    assert observation["after"]["count"] == 1
    assert events[-1]["observation"] == observation
    assert db.optimization_metrics["compaction_duration"] == 3


def test_segment_read_failure_is_not_reported_as_zero(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.list_persistent_segments.side_effect = [RuntimeError("unavailable"), [1]]
    db._optimize()
    observation = db.optimization_metrics["compaction_segments"][0]
    assert observation["before"]["status"] == "unavailable"
    assert observation["before"]["count"] is None
    assert observation["after"]["count"] == 1
    assert records(capfd.readouterr().out)[-1]["state"] == "segments"


def test_concurrency_events_follow_worker_boundaries(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    manager = MagicMock()
    manager.__enter__.return_value.Queue.return_value.qsize.return_value = 100
    future = MagicMock()
    future.result.return_value = (10, 0, {"count": 1, "p99": 0.1, "p95": 0.1, "p50": 0.1, "avg": 0.1})
    executor = MagicMock()
    executor.__enter__.return_value.submit.return_value = future
    runner = MultiProcessingSearchRunner.__new__(MultiProcessingSearchRunner)
    runner.concurrencies, runner.filters, runner.test_data = [1, 4], None, [[1.0]]
    with (
        patch("vectordb_bench.backend.runner.mp_runner.mp.Manager", return_value=manager),
        patch("vectordb_bench.backend.runner.mp_runner.concurrent.futures.ProcessPoolExecutor", return_value=executor),
        patch.object(runner, "get_mp_context", return_value=None),
    ):
        result = runner._run_by_dur(1)
    events = records(capfd.readouterr().out)
    assert [(event["state"], event["concurrency"]) for event in events] == [
        ("start", 1),
        ("end", 1),
        ("start", 4),
        ("end", 4),
    ]
    assert result[2] == [1, 4]


class ProcessEventDatabase:
    def init(self):
        return nullcontext()

    def optimize(self, data_size: int | None = None):
        with stage("compaction"):
            self.optimization_metrics = {"compaction_status": "completed", "compaction_duration": 1.0}


def test_compaction_events_cross_real_worker_process(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    runner = CaseRunner.model_construct(
        config=SimpleNamespace(db=DB.Milvus),
        ca=SimpleNamespace(dataset=SimpleNamespace(data=SimpleNamespace(size=1)), optimize_timeout=30),
        db=ProcessEventDatabase(),
    )
    runner._optimize()
    events = records(capfd.readouterr().out)
    assert [(event["stage"], event["state"]) for event in events] == [("compaction", "start"), ("compaction", "end")]
    assert runner.optimization_metrics["compaction_duration"] == 1.0
