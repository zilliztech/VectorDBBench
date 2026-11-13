from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.warm.cli import Warm
from vectordb_bench.backend.clients.warm.config import WarmConfig, WarmIndexConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.frontend.components.run_test.caseSelector import dbCaseConfigSetting
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UI_CASE_CLUSTERS,
    UICaseItem,
    UICaseItemCluster,
    get_selectable_case_items,
)
from vectordb_bench.models import CaseConfig, TaskConfig


@pytest.mark.parametrize(
    "case_type",
    [
        CaseType.CapacityDim128,
        CaseType.CapacityDim960,
        CaseType.StreamingPerformanceCase,
        CaseType.StreamingCustomDataset,
    ],
)
def test_unsupported_workloads_rejected_before_client_creation(tmp_path: Path, case_type: CaseType):
    path = tmp_path / "index"
    with pytest.raises(ValidationError, match="streaming and capacity cases are unsupported"):
        TaskConfig(
            db=DB.Warm,
            db_config=WarmConfig(db_label="test", path=str(path)),
            db_case_config=WarmIndexConfig(),
            case_config=CaseConfig(case_id=case_type),
            insert_batch_size=1,
        )
    assert not path.exists()
    result = CliRunner().invoke(Warm, ["--case-type", case_type.name, "--path", str(path)])
    assert result.exit_code != 0
    assert "streaming and capacity cases are unsupported" in str(result.exception)
    assert not path.exists()


def test_ui_hides_unsupported_workloads_for_mixed_selections():
    items = [UICaseItem(cases=[], label=label.name, caseLabel=label) for label in CaseLabel]
    cluster = UICaseItemCluster(label="test", uiCaseItems=items)
    for dbs in ([DB.Warm], [DB.Milvus, DB.Warm]):
        selected = get_selectable_case_items(cluster, dbs)
        assert [item.caseLabel for item in selected] == [CaseLabel.Performance]
        for actual_cluster in UI_CASE_CLUSTERS:
            assert all(
                case.case.label == CaseLabel.Performance
                for item in get_selectable_case_items(actual_cluster, dbs)
                for case in item.cases
            )
    assert get_selectable_case_items(cluster, [DB.Milvus]) == items


@pytest.mark.parametrize(
    "values",
    [
        {},
        {
            "m": 24,
            "ef_construction": 400,
            "ef_search": 256,
            "rerank": 200,
            "code_bits": 8,
            "residual_bits": 4,
            "threads": 2,
        },
    ],
)
def test_ui_inputs_reach_task_and_dataset_identity(tmp_path: Path, values: dict):
    case = CaseConfig(case_id=CaseType.Performance768D1M)
    item = UICaseItem(cases=[case])
    container = Mock()
    container.columns.side_effect = lambda count: [container] * count
    seen = set()

    def number_input(label, **kwargs):
        seen.add(label)
        value = values.get(label, kwargs["value"])
        assert kwargs["min_value"] <= value <= kwargs["max_value"]
        return value

    def selectbox(label, options, **kwargs):
        seen.add(label)
        value = values.get(label, options[0])
        assert value in options
        return value

    container.number_input.side_effect = number_input
    container.selectbox.side_effect = selectbox
    fields = {}
    dbCaseConfigSetting(container, {DB.Warm: {item: fields}}, item, [DB.Warm])
    assert seen == {"m", "ef_construction", "ef_search", "rerank", "code_bits", "residual_bits", "threads"}
    container.write.assert_not_called()
    (task,) = generate_tasks(
        [DB.Warm], {DB.Warm: WarmConfig(db_label="test", path=str(tmp_path))}, [case], {DB.Warm: {case: fields}}, 1
    )
    assert task.db_case_config == WarmIndexConfig(**values)
    runner = CaseRunner(
        run_id="test", config=task, ca=case.case, status=RunningStatus.PENDING, dataset_source=DatasetSource.S3
    )
    runner.init_db()
    assert runner.db._identity["dataset"] == case.case.dataset.data.model_dump(mode="json")


@pytest.mark.parametrize(
    "name,minimum", [("m", 1), ("ef_construction", 1), ("ef_search", 1), ("rerank", 0), ("threads", 0)]
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_numeric_parameter_boundaries(name: str, minimum: int, offset: int):
    if offset < 0:
        with pytest.raises(ValidationError):
            WarmIndexConfig(**{name: minimum + offset})
    else:
        assert getattr(WarmIndexConfig(**{name: minimum + offset}), name) == minimum + offset


@pytest.mark.parametrize("name,allowed", [("code_bits", {4, 8}), ("residual_bits", {0, 4, 8})])
@pytest.mark.parametrize("value", [-1, 0, 1, 3, 4, 5, 7, 8, 9])
def test_code_width_boundaries(name: str, allowed: set[int], value: int):
    if value in allowed:
        assert getattr(WarmIndexConfig(**{name: value}), name) == value
    else:
        with pytest.raises(ValidationError):
            WarmIndexConfig(**{name: value})
