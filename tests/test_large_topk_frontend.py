import pytest
from pydantic import ValidationError

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.frontend.components.check_results import charts, data
from vectordb_bench.frontend.components.qps_recall import data as qps_recall_data
from vectordb_bench.frontend.components.run_test.caseSelector import payloadProfileSetting
from vectordb_bench.frontend.components.tables import data as table_data
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UICaseItem,
    generate_normal_cases,
    get_payload_profile_options,
)
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig


def test_performance_ui_case_expands_selected_payload_profiles():
    item = UICaseItem(cases=generate_normal_cases(CaseType.Performance768D100M))
    item.payload_profiles = [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]

    cases = item.get_cases()

    assert [case.payload_profile for case in cases] == [
        PayloadProfile.IDS_ONLY,
        PayloadProfile.VECTOR,
    ]
    assert all(case.case_id == CaseType.Performance768D100M for case in cases)


def test_capacity_ui_case_does_not_expand_payload_profiles():
    item = UICaseItem(cases=generate_normal_cases(CaseType.CapacityDim128))
    item.payload_profiles = [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]

    cases = item.get_cases()

    assert len(cases) == 1
    assert cases[0].payload_profile is None


def test_ui_case_expansion_preserves_payload_conflict_validation():
    item = UICaseItem(
        cases=[
            CaseConfig(
                case_id=CaseType.CloudPayloadSearchCase,
                custom_case={"payload_profile": "vector"},
            )
        ]
    )
    item.payload_profiles = [PayloadProfile.IDS_ONLY]

    with pytest.raises(ValidationError, match="conflicts with custom_case"):
        item.get_cases()


def test_payload_profile_options_require_only_supported_backends():
    assert get_payload_profile_options([DB.Milvus]) == [
        PayloadProfile.IDS_ONLY,
        PayloadProfile.VECTOR,
    ]
    assert get_payload_profile_options([DB.Milvus, DB.ZillizCloud]) == [
        PayloadProfile.IDS_ONLY,
        PayloadProfile.VECTOR,
    ]
    assert get_payload_profile_options([DB.Milvus, DB.Test]) == [PayloadProfile.IDS_ONLY]
    assert get_payload_profile_options([]) == [PayloadProfile.IDS_ONLY]


def test_payload_profile_setting_records_frontend_selection():
    class FakeContainer:
        def __init__(self):
            self.options = []

        def multiselect(self, label, options, default, format_func, key):
            assert label == "Return scenario"
            assert default == [PayloadProfile.IDS_ONLY]
            assert format_func(PayloadProfile.VECTOR) == "Vector payload"
            assert key
            self.options = options
            return options

        def error(self, message):
            raise AssertionError(message)

    item = UICaseItem(cases=generate_normal_cases(CaseType.Performance768D100M))
    container = FakeContainer()

    payloadProfileSetting(container, item, [DB.Milvus])

    assert container.options == [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]
    assert item.payload_profiles == [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]


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


def test_merge_tasks_keeps_payload_profiles_separate_for_same_k():
    merged, failed = data.mergeTasks(
        [
            _case_result(k=1_000_000, qps=10, payload_profile=PayloadProfile.IDS_ONLY),
            _case_result(k=1_000_000, qps=5, payload_profile=PayloadProfile.VECTOR),
        ]
    )

    assert failed == {}
    assert len(merged) == 2
    assert {item["payload_profile"] for item in merged} == {"ids_only", "vector"}
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


def _case_result(
    *,
    k: int,
    qps: float,
    payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
) -> CaseResult:
    return CaseResult(
        task_config=TaskConfig(
            db=DB.Test,
            db_config=DB.Test.config_cls(db_label="same-db"),
            db_case_config=EmptyDBCaseConfig(),
            case_config=CaseConfig(
                case_id=CaseType.Performance768D100M,
                k=k,
                payload_profile=payload_profile,
            ),
        ),
        metrics=Metric(qps=qps, payload_profile=payload_profile.value),
    )
