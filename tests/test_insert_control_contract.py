import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vectordb_bench import config
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig
from vectordb_bench.backend.clients.test.config import TestConfig
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig, TestResult


def make_task(
    case_id: CaseType = CaseType.Performance768D1M,
    custom_case: dict | None = None,
    **overrides: Any,
) -> TaskConfig:
    values = {
        "db": DB.Test,
        "db_config": TestConfig(),
        "db_case_config": EmptyDBCaseConfig(),
        "case_config": CaseConfig(case_id=case_id, custom_case=custom_case),
    }
    values.update(overrides)
    return TaskConfig(**values)


def write_legacy_result(tmp_path: Path, task: TaskConfig, metrics: Metric) -> Path:
    result = TestResult(
        run_id="legacy-run",
        task_label="legacy",
        results=[CaseResult(task_config=task, metrics=metrics)],
    )
    raw = result.model_dump(mode="json", serialize_as_any=True)
    raw["results"][0]["task_config"].pop("insert_batch_size")
    result_path = tmp_path / "legacy.json"
    result_path.write_text(json.dumps(raw))
    return result_path


def test_insert_control_defaults_and_serialization():
    task = make_task()

    assert config.DEFAULT_INSERT_BATCH_SIZE == 100
    assert config.DEFAULT_STREAMING_INSERT_RATE == 500
    assert task.insert_batch_size == 100
    assert task.model_dump(mode="json")["insert_batch_size"] == 100
    assert "insert_rate" not in TaskConfig.model_fields


@pytest.mark.parametrize("insert_batch_size", [0, -1])
def test_insert_batch_size_must_be_positive(insert_batch_size: int):
    with pytest.raises(ValidationError, match="greater than 0"):
        make_task(insert_batch_size=insert_batch_size)


@pytest.mark.parametrize(
    "case_id",
    [CaseType.StreamingPerformanceCase, CaseType.StreamingCustomDataset],
)
def test_streaming_rate_must_cover_and_divide_batch(case_id: CaseType):
    valid = make_task(case_id, {"insert_rate": 1000}, insert_batch_size=250)
    assert valid.case_config.custom_case["insert_rate"] == 1000

    with pytest.raises(ValidationError, match="greater than or equal"):
        make_task(case_id, {"insert_rate": 200}, insert_batch_size=250)
    with pytest.raises(ValidationError, match="divisible"):
        make_task(case_id, {"insert_rate": 550}, insert_batch_size=100)


def test_streaming_rate_uses_stable_default():
    assert make_task(CaseType.StreamingPerformanceCase, insert_batch_size=250).insert_batch_size == 250
    with pytest.raises(ValidationError, match="divisible"):
        make_task(CaseType.StreamingPerformanceCase, insert_batch_size=300)


def test_read_file_migrates_cloud_insert_batch_size(tmp_path: Path):
    path = write_legacy_result(
        tmp_path,
        make_task(
            CaseType.CloudInsertCase,
            {"batch_size": 250},
            insert_batch_size=100,
        ),
        Metric(additional_parameters={"num_per_batch": 100}),
    )

    result = TestResult.read_file(path)

    assert result.results[0].task_config.insert_batch_size == 250


def test_read_file_migrates_metrics_batch_size(tmp_path: Path):
    path = write_legacy_result(
        tmp_path,
        make_task(
            CaseType.StreamingPerformanceCase,
            {"insert_rate": 500},
            insert_batch_size=100,
        ),
        Metric(additional_parameters={"num_per_batch": 250}),
    )

    result = TestResult.read_file(path)

    assert result.results[0].task_config.insert_batch_size == 250


def test_rest_run_accepts_insert_batch_size(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("flask")
    restful_app = importlib.import_module("vectordb_bench.restful.app")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(restful_app.benchmark_runner, "has_running", lambda: False)
    monkeypatch.setattr(restful_app.benchmark_runner, "set_download_address", lambda _value: None)
    monkeypatch.setattr(
        restful_app.benchmark_runner,
        "run",
        lambda tasks, task_label: captured.update(tasks=tasks, task_label=task_label),
    )

    response = restful_app.app.test_client().post(
        "/run",
        json={
            "task_label": "contract",
            "tasks": [
                {
                    "db": DB.Test.value,
                    "db_config": {},
                    "db_case_config": {},
                    "case_config": {
                        "case_id": CaseType.StreamingPerformanceCase.value,
                        "custom_case": {"insert_rate": 1000},
                    },
                    "stages": [],
                    "insert_batch_size": 250,
                }
            ],
        },
    )

    assert response.get_json()["code"] == 0
    assert captured["task_label"] == "contract"
    assert captured["tasks"][0].insert_batch_size == 250
