from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.frontend.components.check_results import charts, data
from vectordb_bench.frontend.components.qps_recall import data as qps_recall_data
from vectordb_bench.frontend.components.tables import data as table_data
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig


def test_merge_tasks_keeps_results_with_different_k_separate():
    merged, failed = data.mergeTasks(
        [
            _case_result(k=100, qps=10),
            _case_result(k=1_000_000, qps=5),
        ]
    )

    assert failed == {}
    assert len(merged) == 2
    assert {item["k"] for item in merged} == {100, 1_000_000}
    assert len({item["case_name"] for item in merged}) == 2


def test_build_recall_at_chart_data_normalizes_and_sorts_cutoffs():
    assert hasattr(charts, "buildRecallAtChartData")
    chart_data = charts.buildRecallAtChartData(
        [
            {
                "db": "milvus",
                "db_name": "milvus-flat",
                "recall_at": {"1000": 0.8, 100: 0.9},
            }
        ]
    )

    assert chart_data == [
        {"k": 100, "recall": 0.9, "db": "milvus", "db_name": "milvus-flat"},
        {"k": 1_000, "recall": 0.8, "db": "milvus", "db_name": "milvus-flat"},
    ]


def test_qps_recall_data_uses_k_aware_case_name():
    task = _case_result(k=1_000_000, qps=5)
    case_name = data.getCaseResultName(task)

    chart_data, failed = qps_recall_data.getChartData(
        [task],
        dbNames=[task.task_config.db_name],
        caseNames=[case_name],
    )

    assert failed == {}
    assert chart_data[0]["case_name"] == case_name
    assert chart_data[0]["k"] == 1_000_000


def test_results_table_uses_k_aware_case_name():
    rows = table_data.formatData(
        [
            _case_result(k=100, qps=10),
            _case_result(k=1_000_000, qps=5),
        ]
    )

    assert [row["k"] for row in rows] == [100, 1_000_000]
    assert len({row["case_name"] for row in rows}) == 2


def _case_result(*, k: int, qps: float) -> CaseResult:
    return CaseResult(
        task_config=TaskConfig(
            db=DB.Test,
            db_config=DB.Test.config_cls(db_label="same-db"),
            db_case_config=EmptyDBCaseConfig(),
            case_config=CaseConfig(case_id=CaseType.Performance768D100M, k=k),
        ),
        metrics=Metric(qps=qps),
    )
