from collections import defaultdict

import pytest

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.frontend.components.run_test import generateTasks
from vectordb_bench.frontend.components.run_test.runSettings import (
    DEFAULT_STREAMING_INSERT_RATE,
    validate_streaming_insert_rates,
)
from vectordb_bench.frontend.components.run_test.submitTask import (
    advancedSettings,
    apply_run_settings,
    get_max_search_k,
)
from vectordb_bench.frontend.config.dbCaseConfigs import custom_streaming_config
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def streaming_case(insert_rate: int | None = None) -> CaseConfig:
    custom_case = {} if insert_rate is None else {"insert_rate": insert_rate}
    return CaseConfig(case_id=CaseType.StreamingPerformanceCase, custom_case=custom_case)


def laion_task(filter_rate: float | None = None) -> TaskConfig:
    if filter_rate is None:
        case_config = CaseConfig(case_id=CaseType.Performance768D100M)
    else:
        case_config = CaseConfig(
            case_id=CaseType.NewIntFilterPerformanceCase,
            custom_case={
                "dataset_with_size_type": DatasetWithSizeType.LAIONLarge,
                "filter_rate": filter_rate,
            },
        )
    return TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=case_config,
    )


def fts_task() -> TaskConfig:
    return TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(case_id=CaseType.FTSBm25Performance),
    )


class SettingsContainer:
    def __init__(self):
        self.number_inputs = {}

    def columns(self, _widths):
        return [self, self]

    def checkbox(self, _label, *, value):
        return value

    def caption(self, _text):
        return None

    def number_input(self, label, **kwargs):
        self.number_inputs[label] = kwargs
        return kwargs["value"]

    def text_input(self, _label, *, value, **_kwargs):
        return value


def test_global_k_control_uses_most_restrictive_selected_case():
    tasks = [laion_task(), laion_task(0.995), laion_task(0.999), fts_task()]
    container = SettingsContainer()

    assert get_max_search_k([laion_task()]) == 1_000_000
    assert get_max_search_k([laion_task(0.995)]) == 500_000
    assert get_max_search_k([laion_task(0.998)]) == 200_000
    assert get_max_search_k(tasks) == 100_000
    advancedSettings(container, tasks)

    assert container.number_inputs["k"]["max_value"] == 100_000


def test_global_k_control_leaves_fts_uncapped():
    container = SettingsContainer()

    assert get_max_search_k([fts_task()]) is None
    advancedSettings(container, [fts_task()])

    assert "max_value" not in container.number_inputs["k"]


def test_apply_run_settings_validates_all_cases_before_mutating_tasks():
    tasks = [laion_task(), laion_task(0.999)]

    with pytest.raises(ValueError, match="supports K up to 100,000"):
        apply_run_settings(
            tasks,
            k=100_001,
            concurrencies=[40, 60],
            concurrency_duration=120,
            load_concurrency=5,
        )

    assert [task.case_config.k for task in tasks] == [100, 100]


def test_apply_run_settings_applies_valid_global_settings():
    tasks = [laion_task(0.999), fts_task()]

    apply_run_settings(
        tasks,
        k=100_000,
        concurrencies=[40, 60],
        concurrency_duration=120,
        load_concurrency=5,
    )

    assert [task.case_config.k for task in tasks] == [100_000, 100_000]
    assert [task.case_config.concurrency_search_config.num_concurrency for task in tasks] == [
        [40, 60],
        [40, 60],
    ]
    assert [task.case_config.concurrency_search_config.concurrency_duration for task in tasks] == [
        120,
        120,
    ]
    assert [task.load_concurrency for task in tasks] == [5, 5]


@pytest.mark.parametrize(
    ("insert_rate", "batch_size", "expected_message"),
    [
        (400, 500, "must be greater than or equal to"),
        (750, 500, "must be divisible by"),
    ],
)
def test_validate_streaming_insert_rates_rejects_invalid_rate(
    insert_rate: int,
    batch_size: int,
    expected_message: str,
):
    is_valid, errors = validate_streaming_insert_rates([streaming_case(insert_rate)], batch_size)

    assert not is_valid
    assert len(errors) == 1
    assert expected_message in errors[0]


def test_validate_streaming_insert_rates_checks_each_streaming_case_and_uses_default():
    cases = [
        CaseConfig(case_id=CaseType.Performance768D1M),
        streaming_case(),
        CaseConfig(case_id=CaseType.StreamingCustomDataset, custom_case={"insert_rate": 1_000}),
    ]

    is_valid, errors = validate_streaming_insert_rates(cases, DEFAULT_STREAMING_INSERT_RATE)

    assert is_valid
    assert errors == []


def test_streaming_rate_control_accepts_valid_values_below_100():
    rate_input = next(item for item in custom_streaming_config if item.label == CaseConfigParamType.insert_rate)

    is_valid, errors = validate_streaming_insert_rates([streaming_case(50)], batch_size=10)

    assert rate_input.inputConfig["min"] == 1
    assert is_valid
    assert errors == []


def test_generate_tasks_passes_batch_size_to_task_config(monkeypatch: pytest.MonkeyPatch):
    captured_task_configs: list[dict[str, object]] = []

    class CapturedTaskConfig:
        def __init__(self, **kwargs: object):
            captured_task_configs.append(kwargs)

    monkeypatch.setattr(generateTasks, "TaskConfig", CapturedTaskConfig)
    case = CaseConfig(case_id=CaseType.Performance768D1M)
    all_case_configs = defaultdict(lambda: defaultdict(dict))

    tasks = generateTasks.generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        [case],
        all_case_configs,
        batch_size=250,
    )

    assert len(tasks) == 1
    assert captured_task_configs[0]["insert_batch_size"] == 250
