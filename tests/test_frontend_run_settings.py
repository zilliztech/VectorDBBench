from collections import defaultdict

import pytest

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.components.run_test import generateTasks
from vectordb_bench.frontend.components.run_test.runSettings import (
    DEFAULT_STREAMING_INSERT_RATE,
    validate_streaming_insert_rates,
)
from vectordb_bench.models import CaseConfig


def streaming_case(insert_rate: int | None = None) -> CaseConfig:
    custom_case = {} if insert_rate is None else {"insert_rate": insert_rate}
    return CaseConfig(case_id=CaseType.StreamingPerformanceCase, custom_case=custom_case)


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
