from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_milvus import TestMilvusOptimize as OptimizeFixture
from vectordb_bench.backend.clients import DB, MetricType
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.interface import dataset_metadata
from vectordb_bench.metric import Metric


class ProcessDatabase:
    def init(self):
        return nullcontext()

    def optimize(self, data_size=None):
        self.optimization_metrics = {"compaction_status": "completed", "compaction_duration": 12.5}


def test_compaction_times_only_requests_and_completion_waits():
    db = OptimizeFixture()._milvus()
    db.client.compact.side_effect = [41, 42]
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.perf_counter", side_effect=[10, 13, 100, 105]):
        db._optimize()
    assert db.optimization_metrics == {"compaction_status": "completed", "compaction_duration": 8.0}
    # The 87-second readiness interval is excluded; the two request/wait intervals are summed.
    assert db._wait_for_compaction.call_count == 2


@pytest.mark.parametrize("scenario,status", [("gpu", "skipped"), ("no_plan", "skipped"), ("denied", "failed")])
def test_compaction_not_measured_is_null(scenario, status):
    db = OptimizeFixture()._milvus(is_gpu_index=scenario == "gpu", force_merge_enabled=False)
    if scenario == "no_plan":
        db.client.compact.return_value = 0
    if scenario == "denied":
        error = RuntimeError("denied")
        error.code = SimpleNamespace(name="PERMISSION_DENIED")
        db.client.compact.side_effect = error
    db._optimize()
    assert db.optimization_metrics == {"compaction_status": status, "compaction_duration": None}


def test_compaction_failure_does_not_publish_partial_duration():
    db = OptimizeFixture()._milvus(compact_side_effect=RuntimeError("failed"))
    with pytest.raises(RuntimeError, match="failed"):
        db._optimize()
    assert db.optimization_metrics == {"compaction_status": "failed", "compaction_duration": None}


def test_optimization_observation_crosses_the_worker_process_boundary():
    runner = CaseRunner.model_construct(
        config=SimpleNamespace(db=DB.Milvus),
        ca=SimpleNamespace(dataset=SimpleNamespace(data=SimpleNamespace(size=1)), optimize_timeout=30),
        db=ProcessDatabase(),
    )
    assert runner._optimize() >= 0
    assert runner.optimization_metrics == {"compaction_status": "completed", "compaction_duration": 12.5}
    assert not hasattr(runner.db, "optimization_metrics")  # Child mutations do not reach the parent.


def test_builtin_dataset_metadata_and_legacy_metric_defaults():
    runner = SimpleNamespace(
        dataset_source=DatasetSource.S3,
        ca=SimpleNamespace(dataset=SimpleNamespace(data=SimpleNamespace(
            name="Cohere", size=1_000_000, dim=768, metric_type=MetricType.COSINE,
        ))),
    )
    metadata = dataset_metadata(runner).model_dump(mode="json")
    assert metadata["size"] == 1_000_000
    assert metadata["dim"] == 768
    assert metadata["source"] == "S3"
    assert metadata["revision"] is None
    assert asdict(Metric(insert_duration=2))["compaction_duration"] is None
    assert Metric().compaction_status == "not_recorded"


def test_resolved_output_fields_match_payload_profiles():
    db = OptimizeFixture()._milvus()
    db._vector_field, db._scalar_label_field = "vector", "label"
    assert db.search_output_fields(PayloadProfile.IDS_ONLY) == []
    assert db.search_output_fields(PayloadProfile.VECTOR) == ["vector"]
    assert db.search_output_fields(PayloadProfile.SCALAR_LABEL) == ["label"]


def test_saved_native_result_retains_observations_and_reads_legacy_json(tmp_path):
    import json
    from vectordb_bench.backend.cases import CaseType
    from vectordb_bench.backend.clients.milvus.config import AutoIndexConfig, MilvusConfig
    from vectordb_bench.interface import BenchMarkRunner
    from vectordb_bench.models import CaseConfig, TaskConfig, TaskStage, TestResult
    from unittest.mock import MagicMock

    config = TaskConfig(db=DB.Milvus, db_config=MilvusConfig(uri="http://example.invalid"),
                        db_case_config=AutoIndexConfig(), case_config=CaseConfig(case_id=CaseType.Performance768D1M),
                        stages=[TaskStage.LOAD, TaskStage.DROP_OLD])
    data = SimpleNamespace(name="Cohere", size=1_000_000, dim=768, metric_type=MetricType.COSINE)
    runner = SimpleNamespace(config=config, ca=SimpleNamespace(dataset=SimpleNamespace(data=data), payload_profile=PayloadProfile.IDS_ONLY),
                             dataset_source=DatasetSource.S3, optimization_metrics={"compaction_status": "completed", "compaction_duration": 8.0},
                             db=SimpleNamespace(search_output_fields=lambda _: []), is_fts=False,
                             load_reuse_key=lambda: None, display=lambda: "fixture", run=lambda _: Metric(insert_duration=2))
    task = SimpleNamespace(case_runners=[runner], num_cases=lambda: 1, task_label="test", run_id="test")
    with patch.object(TestResult, "flush", autospec=True) as flush, patch.object(TestResult, "display"):
        BenchMarkRunner()._async_task_v2(task, MagicMock())
    result = flush.call_args.args[0]
    doc = result.model_dump_for_output()
    assert doc["results"][0]["metrics"]["compaction_duration"] == 8
    assert doc["results"][0]["metrics"]["output_fields"] == []
    assert doc["results"][0]["dataset_metadata"]["size"] == 1_000_000
    # Existing results without additive fields remain loadable through the native reader.
    for key in ["compaction_duration", "compaction_status", "output_fields"]:
        doc["results"][0]["metrics"].pop(key)
    path = tmp_path / "result_old.json"
    path.write_text(json.dumps(doc))
    old = TestResult.read_file(path)
    assert old.results[0].metrics.compaction_duration is None
    assert old.results[0].metrics.compaction_status == "not_recorded"
