from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.models import CaseConfig, TaskConfig, TaskStage


class SearchProbeDB:
    name = "SearchProbeDB"

    def __init__(self, results=None):
        self.results = list(results or [])
        self.init_calls = 0
        self.search_calls = 0

    def supports_payload_profile(self, payload_profile):
        return True

    @contextmanager
    def init(self):
        self.init_calls += 1
        yield

    def prepare_filter(self, filters):
        return None

    def search_embedding(self, query, k=100, payload_profile=None, tenant=None):
        self.search_calls += 1
        if self.results:
            return self.results.pop(0)
        return []


def _large_topk_property_runner(db: DB, k: int) -> CaseRunner:
    return CaseRunner.model_construct(
        config=SimpleNamespace(db=db, case_config=SimpleNamespace(k=k)),
        ca=SimpleNamespace(label=CaseLabel.Performance),
    )


@pytest.mark.parametrize("db", [DB.Milvus, DB.ZillizCloud])
def test_large_topk_selects_collection_mode_and_logs_requested_k(db, monkeypatch):
    runner = _large_topk_property_runner(db, 100_000)
    messages = []
    monkeypatch.setattr(
        "vectordb_bench.backend.task_runner.log.info",
        lambda message, *args: messages.append(message % args),
    )

    properties = runner._collection_properties(log_selection=True)

    assert properties == {"query_mode": "large_topk"}
    assert db.value in messages[0]
    assert "requested K=100000" in messages[0]
    assert "query_mode=large_topk" in messages[0]


@pytest.mark.parametrize(
    ("db", "k"),
    [
        # Milvus and Zilliz Cloud stay in default mode at or below the TopK limit.
        (DB.ZillizCloud, 16_384),
        (DB.Milvus, 16_384),
        (DB.Milvus, 100),
        # Backends without a Large TopK collection mode are never reconfigured.
        (DB.Pinecone, 100_000),
    ],
)
def test_large_topk_collection_mode_does_not_change_other_workloads(db, k):
    runner = _large_topk_property_runner(db, k)

    assert runner._collection_properties() == {}


def test_serial_runner_rejects_query_count_mismatch_before_db_init():
    db = SearchProbeDB()
    runner = SerialSearchRunner(
        db=db,
        test_data=[[0.1], [0.2]],
        ground_truth=[[1, 2, 3, 4]],
        k=4,
    )

    with pytest.raises(ValueError, match="query count"):
        runner.search((runner.test_data, runner.ground_truth))

    assert db.init_calls == 0
    assert db.search_calls == 0


def test_serial_runner_rejects_narrow_ground_truth_before_db_init():
    db = SearchProbeDB()
    runner = SerialSearchRunner(
        db=db,
        test_data=[[0.1]],
        ground_truth=[[1, 2]],
        k=4,
    )

    with pytest.raises(ValueError, match="width"):
        runner.search((runner.test_data, runner.ground_truth))

    assert db.init_calls == 0
    assert db.search_calls == 0


def test_serial_runner_reports_p50_and_prefix_correct_recall():
    ground_truth = list(range(1_000))
    results = [*range(50), *range(100, 150), *range(50, 100), *range(150, 1_000)]
    db = SearchProbeDB(results=[results])
    runner = SerialSearchRunner(
        db=db,
        test_data=[[0.1]],
        ground_truth=[ground_truth],
        k=1_000,
    )

    recall, ndcg, p99, p95, p50, recall_at = runner.search((runner.test_data, runner.ground_truth))

    assert recall == 1.0
    assert ndcg == 1.0
    assert p99 >= p50 >= 0
    assert p95 >= p50
    assert recall_at == {100: 0.5, 1_000: 1.0}


def test_concurrent_latency_aggregation_includes_p50():
    runner = MultiProcessingSearchRunner(db=SearchProbeDB(), test_data=[[0.1]])
    results = [
        (10, 0, {"p99": 0.9, "p95": 0.8, "p50": 0.3, "avg": 0.4, "count": 10}),
        (20, 0, {"p99": 0.7, "p95": 0.6, "p50": 0.2, "avg": 0.3, "count": 20}),
    ]

    assert runner._aggregate_latency_stats(results) == pytest.approx((0.9, 0.8, 0.3, 1 / 3))


def test_concurrent_latency_aggregation_handles_empty_success_window():
    runner = MultiProcessingSearchRunner(db=SearchProbeDB(), test_data=[[0.1]])

    assert runner._aggregate_latency_stats([]) == (0, 0, 0, 0)
    assert hasattr(runner, "_latency_summary")
    assert runner._latency_summary([]) == (0, 0, 0, 0)


def test_case_runner_rejects_unsupported_laion_k_before_db_init(monkeypatch):
    case_config = CaseConfig(case_id=CaseType.Performance768D100M, k=1_000_001)
    runner = CaseRunner(
        run_id="large-topk",
        config=TaskConfig(
            db=DB.Test,
            db_config=DB.Test.config_cls(),
            db_case_config=EmptyDBCaseConfig(),
            case_config=case_config,
        ),
        ca=case_config.case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    init_called = False

    def fake_init_db(self, drop_old=True):
        nonlocal init_called
        init_called = True

    def fake_prepare(self, *args, **kwargs):
        return self.resolve_search_files(k=case_config.k, filters=kwargs["filters"])

    monkeypatch.setattr(CaseRunner, "init_db", fake_init_db)
    monkeypatch.setattr(DatasetManager, "prepare", fake_prepare)

    with pytest.raises(ValueError, match="LAION"):
        runner._pre_run(drop_old=False)

    assert init_called is False


def test_cloud_cold_latency_keeps_standard_query_artifacts_for_large_k(monkeypatch):
    case_config = CaseConfig(
        case_id=CaseType.CloudColdLatencyCase,
        custom_case={"query_count": 1_000},
        k=1_001,
    )
    runner = CaseRunner(
        run_id="large-topk",
        config=TaskConfig(
            db=DB.Test,
            db_config=DB.Test.config_cls(),
            db_case_config=EmptyDBCaseConfig(),
            case_config=case_config,
            stages=[TaskStage.SEARCH_SERIAL],
        ),
        ca=case_config.case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    selected_files = None

    def fake_init_db(self, drop_old=True):
        return None

    def fake_prepare(self, *args, **kwargs):
        nonlocal selected_files
        selected_files = self.resolve_search_files(k=kwargs["k"], filters=kwargs["filters"])
        return True

    monkeypatch.setattr(CaseRunner, "init_db", fake_init_db)
    monkeypatch.setattr(DatasetManager, "prepare", fake_prepare)

    runner._pre_run(drop_old=False)

    assert selected_files is not None
    assert selected_files.test_file == "test.parquet"
    assert selected_files.query_count == 1_000
    assert runner.config.case_config.k == 1_001


def test_case_runner_propagates_large_topk_metrics(monkeypatch):
    case_config = CaseConfig(case_id=CaseType.Performance768D100M, k=1_000)
    runner = CaseRunner(
        run_id="large-topk",
        config=TaskConfig(
            db=DB.Test,
            db_config=DB.Test.config_cls(),
            db_case_config=EmptyDBCaseConfig(),
            case_config=case_config,
            stages=[TaskStage.SEARCH_CONCURRENT, TaskStage.SEARCH_SERIAL],
        ),
        ca=case_config.case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )

    class SerialRunner:
        def run(self):
            return (0.8, 0.7, 0.9, 0.8, 0.5, {100: 0.9, 1_000: 0.8}), 0.1

    class ConcurrentRunner:
        def run(self):
            return 12.0, [1], [12.0], [0.9], [0.8], [0.6], [0.5]

        def stop(self):
            return None

    def fake_init_search_runners(self):
        self.serial_search_runner = SerialRunner()
        self.search_runner = ConcurrentRunner()

    monkeypatch.setattr(CaseRunner, "_init_search_runners", fake_init_search_runners)
    runner.ca.dataset.gt_data = SimpleNamespace(
        path=Path("neighbors_top100k_nq200.parquet"),
        row_count=200,
        width=100_000,
    )

    metrics = runner._run_perf_case(drop_old=False)

    assert metrics.serial_latency_p50 == 0.5
    assert metrics.conc_latency_p50_list == [0.5]
    assert metrics.recall_at == {100: 0.9, 1_000: 0.8}
    assert metrics.additional_parameters["ground_truth"] == {
        "file": "neighbors_top100k_nq200.parquet",
        "query_count": 200,
        "width": 100_000,
    }
