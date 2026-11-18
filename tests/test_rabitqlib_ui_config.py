"""Offline tests for the RaBitQLib web UI case config.

The UI builds a case config from the inputs `CASE_CONFIG_MAP` declares for a
database; without an entry it shows "Auto" and runs with defaults only. The
case-support tests cover the UI and direct task construction; the CLI's own
check is tested in test_rabitqlib.py.
These tests need neither rabitqlib nor a browser.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.rabitqlib.cli import RaBitQLibHNSW
from vectordb_bench.backend.clients.rabitqlib.config import RaBitQLibConfig, RaBitQLibHNSWConfig
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    CASE_CONFIG_MAP,
    UI_CASE_CLUSTERS,
    UICaseItem,
    UICaseItemCluster,
    get_case_config_inputs,
    get_custom_case_cluter,
    get_custom_streaming_case_cluster,
    get_selectable_case_items,
)
from vectordb_bench.models import CaseConfig, TaskConfig


def ui_task(values: dict) -> RaBitQLibHNSWConfig:
    """Generate a task the way the Run Test page does, from input label to value."""
    case = CaseConfig(case_id=CaseType.Performance768D1M)
    inputs = get_case_config_inputs(DB.RaBitQLib, CaseLabel.Performance)
    ui_values = {i.label: values.get(i.label.value, i.inputConfig["value"]) for i in inputs}
    [task] = generate_tasks(
        [DB.RaBitQLib],
        {DB.RaBitQLib: RaBitQLibConfig(path="/tmp/unused")},
        [case],
        {DB.RaBitQLib: {case: ui_values}},
        batch_size=100,
    )
    config = task.db_case_config
    config.metric_type = MetricType.COSINE  # the assembler copies the dataset metric in
    return config


def test_ui_exposes_the_tunable_params():
    # seed and build_threads stay at their defaults in the UI.
    labels = {i.displayLabel for i in get_case_config_inputs(DB.RaBitQLib, CaseLabel.Performance)}

    assert labels == {"nbits", "num_clusters", "M", "ef_construction", "ef_search"}


def test_ui_declares_inputs_for_performance_cases_only():
    assert set(CASE_CONFIG_MAP[DB.RaBitQLib]) == {CaseLabel.Performance}


@pytest.mark.parametrize("dbs", [[DB.RaBitQLib], [DB.Milvus, DB.RaBitQLib]])
def test_ui_offers_only_performance_cases(dbs: list[DB]):
    items = [UICaseItem(cases=[], label=label.name, caseLabel=label) for label in CaseLabel]
    cluster = UICaseItemCluster(label="test", uiCaseItems=items)
    assert [item.caseLabel for item in get_selectable_case_items(cluster, dbs)] == [CaseLabel.Performance]

    clusters = [*UI_CASE_CLUSTERS, get_custom_case_cluter(), get_custom_streaming_case_cluster()]
    offered = [case for c in clusters for item in get_selectable_case_items(c, dbs) for case in item.cases]
    assert offered
    assert all(case.case.label == CaseLabel.Performance for case in offered)


@pytest.mark.parametrize(
    "case_type",
    [
        CaseType.CapacityDim128,
        CaseType.CapacityDim960,
        CaseType.StreamingPerformanceCase,
        CaseType.StreamingCustomDataset,
        CaseType.FTSBm25Performance,
        CaseType.CloudInsertCase,
    ],
)
def test_task_construction_rejects_non_performance_cases(tmp_path: Path, case_type: CaseType):
    # Covers UI and programmatic callers: TaskConfig fails before any client exists or file is written.
    path = tmp_path / "index"
    with pytest.raises(ValidationError, match="supports only performance cases"):
        TaskConfig(
            db=DB.RaBitQLib,
            db_config=RaBitQLibConfig(path=str(path)),
            db_case_config=RaBitQLibHNSWConfig(),
            case_config=CaseConfig(case_id=case_type),
        )
    assert not path.exists()


def test_task_construction_accepts_performance_cases(tmp_path: Path):
    TaskConfig(
        db=DB.RaBitQLib,
        db_config=RaBitQLibConfig(path=str(tmp_path)),
        db_case_config=RaBitQLibHNSWConfig(),
        case_config=CaseConfig(case_id=CaseType.Performance768D1M),
    )


def test_ui_defaults_match_the_config_defaults():
    config = ui_task({})
    defaults = RaBitQLibHNSWConfig(metric_type=MetricType.COSINE)

    assert isinstance(config, RaBitQLibHNSWConfig)
    assert config.index_param() == defaults.index_param()
    assert config.search_param() == defaults.search_param()


def test_tuned_values_reach_the_client():
    config = ui_task({"nbits": 5, "num_clusters": 64, "M": 32, "ef_construction": 400, "ef_search": 256})

    assert config.index_param() == {
        "metric": "ip",
        "nbits": 5,
        "num_clusters": 64,
        "M": 32,
        "ef_construction": 400,
        "seed": 100,
    }
    assert config.search_param() == {"ef": 256}


def test_input_ranges_and_defaults_match_the_cli():
    inputs = {i.label.value: i.inputConfig for i in get_case_config_inputs(DB.RaBitQLib, CaseLabel.Performance)}
    # The CLI option for the M parameter is --m.
    options = {("M" if p.name == "m" else p.name): p for p in RaBitQLibHNSW.params}

    for name, ui in inputs.items():
        option = options[name]
        assert (ui["min"], ui["max"], ui["value"]) == (option.type.min, option.type.max, option.default), name
