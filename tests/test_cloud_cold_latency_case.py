import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import CaseLabel, CaseType, CloudColdLatencyCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.clients.pinecone.config import PineconeConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.cold_warm_runner import ColdWarmSearchRunner
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig, TaskStage, TestResult


def test_cloud_cold_latency_case_defaults_to_laion_100m():
    case = CloudColdLatencyCase()

    assert case.case_id == CaseType.CloudColdLatencyCase
    assert case.label == CaseLabel.CloudColdLatency
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000
    assert case.dataset.data.dim == 768
    assert case.payload_profile == PayloadProfile.IDS_ONLY
    assert case.query_count == 1000
    assert case.filters.type == FilterOp.NonFilter


def test_cloud_cold_latency_case_accepts_payload_dataset_and_int_filter():
    case = CloudColdLatencyCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        payload_profile="vector",
        filter_rate=0.9,
        query_count=10,
    )

    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.name == "Cohere"
    assert case.payload_profile == PayloadProfile.VECTOR
    assert case.filter_rate == 0.9
    assert case.query_count == 10
    assert case.filters.type == FilterOp.NumGE


def test_cloud_cold_latency_case_accepts_label_filter():
    case = CloudColdLatencyCase(label_percentage=0.9)

    assert case.label_percentage == 0.9
    assert case.filters.type == FilterOp.StrEqual


def test_cloud_cold_latency_case_rejects_two_filter_types():
    with pytest.raises(ValueError, match="supports only one filter type"):
        CloudColdLatencyCase(filter_rate=0.9, label_percentage=0.9)


def test_cloud_cold_latency_case_rejects_invalid_query_count():
    with pytest.raises(ValueError, match="query_count must be positive"):
        CloudColdLatencyCase(query_count=0)


def test_case_config_builds_cloud_cold_latency_case_from_custom_case():
    case = CaseConfig(
        case_id=CaseType.CloudColdLatencyCase,
        custom_case={
            "payload_profile": "scalar_label",
            "label_percentage": 0.9,
            "query_count": 12,
        },
    ).case

    assert isinstance(case, CloudColdLatencyCase)
    assert case.payload_profile == PayloadProfile.SCALAR_LABEL
    assert case.label_percentage == 0.9
    assert case.query_count == 12


def test_cli_builds_cloud_cold_latency_custom_case_config():
    params = {
        "case_type": "CloudColdLatencyCase",
        "payload_profile": "vector",
        "cloud_filter_rate": 0.9,
        "cloud_label_percentage": None,
        "cloud_cold_query_count": 1000,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }

    assert get_custom_case_config(params) == {
        "payload_profile": "vector",
        "filter_rate": 0.9,
        "query_count": 1000,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }


def test_cli_keeps_cloud_cold_latency_default_dataset_as_laion():
    params = {
        "case_type": "CloudColdLatencyCase",
        "payload_profile": "ids_only",
        "cloud_filter_rate": None,
        "cloud_label_percentage": None,
        "cloud_cold_query_count": 1000,
        "dataset_with_size_type": None,
    }

    custom_case = get_custom_case_config(params)
    case = CaseConfig(case_id=CaseType.CloudColdLatencyCase, custom_case=custom_case).case

    assert custom_case == {
        "payload_profile": "ids_only",
        "query_count": 1000,
    }
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000


def test_cloud_cold_latency_result_file_uses_cold_latency_metrics(tmp_path: Path):
    cold_latency = {
        "cold_stats": {
            "first_query_latency": 0.2,
            "p99_latency": 0.3,
            "p95_latency": 0.25,
            "avg_latency": 0.21,
        },
        "warm_stats": {
            "first_query_latency": 0.1,
            "p99_latency": 0.15,
            "p95_latency": 0.12,
            "avg_latency": 0.11,
        },
        "ratios": {
            "first_query_latency": 2.0,
            "p99_latency": 2.0,
            "p95_latency": 2.0833,
            "avg_latency": 1.9091,
        },
    }
    result = CaseResult(
        task_config=TaskConfig(
            db=DB.Pinecone,
            db_config=PineconeConfig(
                db_label="pinecone_cloud_cold_latency",
                api_key="secret-key",
                index_name="laion100m",
            ),
            db_case_config=EmptyDBCaseConfig(),
            case_config=CaseConfig(
                case_id=CaseType.CloudColdLatencyCase,
                custom_case={"payload_profile": "vector", "query_count": 1000},
            ),
            stages=[TaskStage.SEARCH_SERIAL],
            load_concurrency=0,
        ),
        metrics=Metric(
            insert_duration=0.0,
            optimize_duration=0.0,
            load_duration=0.0,
            payload_profile="vector",
            payload_estimated_bytes_per_query=309200,
            additional_parameters={"cold_latency": cold_latency},
        ),
    )
    test_result = TestResult(run_id="run-id", task_label="cloud_cold_latency_pinecone", results=[result])

    test_result.write_db_file(tmp_path, test_result, "pinecone")

    result_file = next(tmp_path.glob("result_*_pinecone.json"))
    raw_output = result_file.read_text()
    assert raw_output.startswith('{\n  "run_id"')
    written = json.loads(raw_output)
    assert written["results"][0]["metrics"] == {
        "insert_duration": 0.0,
        "optimize_duration": 0.0,
        "load_duration": 0.0,
        "payload_profile": "vector",
        "payload_estimated_bytes_per_query": 309200,
        "cold_latency": cold_latency,
    }
    assert written["results"][0]["task_config"]["db_config"]["api_key"] == "**********"
    assert written["results"][0]["task_config"]["db_config"]["index_name"] == "laion100m"
    assert written["results"][0]["task_config"]["case_config"] == {
        "case_id": 700,
        "custom_case": {"payload_profile": "vector", "query_count": 1000},
    }

    read_back = TestResult.read_file(result_file)
    assert read_back.results[0].task_config.case_config.case_id == CaseType.CloudColdLatencyCase
    assert read_back.results[0].task_config.case_config.custom_case == {
        "payload_profile": "vector",
        "query_count": 1000,
    }
    assert read_back.results[0].metrics.additional_parameters["cold_latency"] == cold_latency

    frontend_read_back = TestResult.read_file(result_file, trans_unit=True)
    assert frontend_read_back.results[0].metrics.additional_parameters["cold_latency"] == cold_latency


class FakeColdWarmDB:
    name = "FakeColdWarmDB"

    def __init__(self, supported_payload_profiles: set[PayloadProfile] | None = None):
        self.supported_payload_profiles = supported_payload_profiles or {PayloadProfile.IDS_ONLY}
        self.calls = []
        self.prepare_filter_calls = []
        self.init_enter_count = 0

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile in self.supported_payload_profiles

    def need_normalize_cosine(self) -> bool:
        return False

    @contextmanager
    def init(self):
        self.init_enter_count += 1
        yield

    def prepare_filter(self, filters: Filter):
        self.prepare_filter_calls.append(filters)

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        self.calls.append((query, k, kwargs))
        return list(range(k))


def test_cold_warm_runner_computes_stats_and_ratios(monkeypatch: pytest.MonkeyPatch):
    db = FakeColdWarmDB()
    # Cold latencies: 0.2, 0.2, 0.2. Warm latencies: 0.1, 0.1, 0.1.
    perf_values = iter([0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9])
    monkeypatch.setattr("vectordb_bench.backend.runner.cold_warm_runner.time.perf_counter", lambda: next(perf_values))

    runner = ColdWarmSearchRunner(
        db=db,
        test_data=[[0.1], [0.2], [0.3]],
        k=3,
        query_count=3,
    )

    result = runner.run()

    assert result == {
        "cold_stats": {
            "first_query_latency": 0.2,
            "p99_latency": 0.2,
            "p95_latency": 0.2,
            "avg_latency": 0.2,
        },
        "warm_stats": {
            "first_query_latency": 0.1,
            "p99_latency": 0.1,
            "p95_latency": 0.1,
            "avg_latency": 0.1,
        },
        "cold_warm_ratio": {
            "first_query_latency_ratio": 2.0,
            "p99_latency_ratio": 2.0,
            "p95_latency_ratio": 2.0,
            "avg_latency_ratio": 2.0,
        },
    }
    assert db.init_enter_count == 1
    assert len(db.prepare_filter_calls) == 1
    assert [call[0] for call in db.calls] == [[0.1], [0.2], [0.3], [0.1], [0.2], [0.3]]


def test_cold_warm_runner_passes_payload_profile_in_both_passes(monkeypatch: pytest.MonkeyPatch):
    db = FakeColdWarmDB(supported_payload_profiles={PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR})
    perf_values = iter([0.0, 0.1, 0.1, 0.2])
    monkeypatch.setattr("vectordb_bench.backend.runner.cold_warm_runner.time.perf_counter", lambda: next(perf_values))

    runner = ColdWarmSearchRunner(
        db=db,
        test_data=[np.array([0.1])],
        k=3,
        payload_profile=PayloadProfile.VECTOR,
        query_count=1,
    )

    runner.run()

    assert db.calls == [
        ([0.1], 3, {"payload_profile": PayloadProfile.VECTOR}),
        ([0.1], 3, {"payload_profile": PayloadProfile.VECTOR}),
    ]


def test_cold_warm_runner_omits_payload_profile_for_ids_only(monkeypatch: pytest.MonkeyPatch):
    db = FakeColdWarmDB()
    perf_values = iter([0.0, 0.1, 0.1, 0.2])
    monkeypatch.setattr("vectordb_bench.backend.runner.cold_warm_runner.time.perf_counter", lambda: next(perf_values))

    runner = ColdWarmSearchRunner(
        db=db,
        test_data=[[0.1]],
        k=3,
        payload_profile=PayloadProfile.IDS_ONLY,
        query_count=1,
    )

    runner.run()

    assert db.calls == [
        ([0.1], 3, {}),
        ([0.1], 3, {}),
    ]


def test_cold_warm_runner_fails_for_unsupported_payload_profile():
    db = FakeColdWarmDB()

    with pytest.raises(NotImplementedError, match="payload_profile=vector"):
        ColdWarmSearchRunner(
            db=db,
            test_data=[[0.1]],
            payload_profile=PayloadProfile.VECTOR,
            query_count=1,
        )


def test_cold_warm_runner_rejects_too_few_queries():
    db = FakeColdWarmDB()

    with pytest.raises(ValueError, match="query_count=2 exceeds test_data size=1"):
        ColdWarmSearchRunner(db=db, test_data=[[0.1]], query_count=2)


def test_assembler_schedules_cloud_cold_latency_case():
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.CloudColdLatencyCase,
            custom_case={"query_count": 1},
        ),
        stages=[TaskStage.SEARCH_SERIAL],
    )

    runner = Assembler.assemble_all("run-id", "task-label", [task], DatasetSource.S3)

    assert len(runner.case_runners) == 1
    assert runner.case_runners[0].ca.label == CaseLabel.CloudColdLatency


def test_case_runner_stores_cloud_cold_latency_metric(monkeypatch: pytest.MonkeyPatch):
    case = CloudColdLatencyCase(query_count=1)
    case.dataset.test_data = [[0.1]]
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.CloudColdLatencyCase,
            custom_case={"query_count": 1},
        ),
        stages=[TaskStage.SEARCH_SERIAL],
    )
    runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    runner.db = FakeColdWarmDB()

    expected = {
        "cold_stats": {
            "first_query_latency": 0.2,
            "p99_latency": 0.2,
            "p95_latency": 0.2,
            "avg_latency": 0.2,
        },
        "warm_stats": {
            "first_query_latency": 0.1,
            "p99_latency": 0.1,
            "p95_latency": 0.1,
            "avg_latency": 0.1,
        },
        "cold_warm_ratio": {
            "first_query_latency_ratio": 2.0,
            "p99_latency_ratio": 2.0,
            "p95_latency_ratio": 2.0,
            "avg_latency_ratio": 2.0,
        },
    }
    captured_kwargs = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self):
            return expected

    monkeypatch.setattr("vectordb_bench.backend.task_runner.ColdWarmSearchRunner", FakeRunner)

    metric = runner._run_cloud_cold_latency_case()

    assert metric.additional_parameters["cold_latency"] == expected
    assert metric.payload_profile == "ids_only"
    assert metric.payload_estimated_bytes_per_query == case.estimated_payload_bytes_per_query(task.case_config.k)
    assert captured_kwargs == {
        "db": runner.db,
        "test_data": [[0.1]],
        "filters": case.filters,
        "k": task.case_config.k,
        "payload_profile": case.payload_profile,
        "query_count": case.query_count,
    }
