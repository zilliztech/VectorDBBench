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
from vectordb_bench.backend.clients.milvus.milvus import Milvus
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


@pytest.mark.parametrize("failure", [False, True])
def test_index_readiness_reports_each_wait_and_failure(monkeypatch, capfd, failure):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db._main_index_name = "vector_index"
    db.client.describe_index.side_effect = [
        {"pending_index_rows": 10},
        RuntimeError("index read failed") if failure else {"pending_index_rows": 0},
    ]
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.sleep") as sleep:
        if failure:
            with pytest.raises(RuntimeError, match="index read failed"):
                Milvus._wait_for_index(db)
        else:
            Milvus._wait_for_index(db)
        sleep.assert_called_once_with(5)
    start, end = records(capfd.readouterr().out)
    assert start["stage"] == end["stage"] == "index"
    assert start["span_id"] == end["span_id"]
    assert end["state"] == ("failed" if failure else "end")
    assert start["index_name"] == db._main_index_name


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


def test_nested_stage_events_link_to_parent_and_restore_context(monkeypatch, capfd):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    with stage("index_compaction") as parent:
        with pytest.raises(ValueError), stage("index"):
            raise ValueError("failed")
        with stage("compaction"):
            pass
    with stage("collection_load"):
        pass
    events = records(capfd.readouterr().out)
    children = [event for event in events if event["stage"] in ("index", "compaction")]
    assert all(event["parent_span_id"] == parent for event in children)
    assert all("parent_span_id" not in event for event in events if event["stage"] == "collection_load")


def test_reload_measures_unload_and_load_after_combined_stage(monkeypatch, capfd):
    from pymilvus.client.types import LoadState

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.get_load_state.side_effect = [{"state": LoadState.NotLoad}, {"state": LoadState.Loaded}]
    db._optimize()
    events = records(capfd.readouterr().out)
    parent = next(event for event in events if event["stage"] == "index_compaction" and event["state"] == "start")
    compaction = next(event for event in events if event["stage"] == "compaction" and event["state"] == "start")
    assert compaction["parent_span_id"] == parent["span_id"]
    unload, load = [
        next(event for event in events if event["stage"] == name and event["state"] == "end")
        for name in ("collection_unload", "collection_load")
    ]
    assert unload["reload_id"] == load["reload_id"]
    assert unload["profile"] is False
    db.client.release_collection.assert_called_once()
    db.client.load_collection.assert_called_once()
    db.client.refresh_load.assert_not_called()


@pytest.mark.parametrize("operation", ["release_collection", "load_collection"])
def test_reload_failure_is_not_reported_as_completed(monkeypatch, capfd, operation):
    from pymilvus.client.types import LoadState

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.get_load_state.return_value = {"state": LoadState.NotLoad}
    getattr(db.client, operation).side_effect = RuntimeError("reload failed")
    with pytest.raises(RuntimeError, match="reload failed"):
        db._optimize()
    events = records(capfd.readouterr().out)
    failed = "collection_unload" if operation == "release_collection" else "collection_load"
    assert any(event["stage"] == failed and event["state"] == "failed" for event in events)
    assert not any(event["stage"] == "collection_load" and event["state"] == "end" for event in events)
    assert db.optimization_metrics["compaction_status"] == "completed"


def test_reload_fails_when_collection_disappears():
    from pymilvus.client.types import LoadState

    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.get_load_state.return_value = {"state": LoadState.NotExist}
    with pytest.raises(RuntimeError, match="disappeared"):
        db._wait_collection_state(LoadState.NotLoad, float("inf"))


def test_reload_state_wait_has_deadline():
    from pymilvus.client.types import LoadState

    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.monotonic", return_value=10):
        with pytest.raises(TimeoutError, match="before timeout"):
            db._wait_collection_state(LoadState.Loaded, 10)
    db.client.get_load_state.assert_not_called()
