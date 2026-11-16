"""Offline tests for the RaBitQLib web UI case config.

The UI builds a case config from the inputs `CASE_CONFIG_MAP` declares for a
database; without an entry it shows "Auto" and runs with defaults only.
These tests need neither rabitqlib nor a browser.
"""

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.rabitqlib.config import RaBitQLibConfig, RaBitQLibHNSWConfig
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import CASE_CONFIG_MAP, get_case_config_inputs
from vectordb_bench.models import CaseConfig


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


def test_ui_exposes_every_build_and_search_param():
    labels = {i.displayLabel for i in get_case_config_inputs(DB.RaBitQLib, CaseLabel.Performance)}

    assert labels == {"nbits", "num_clusters", "M", "ef_construction", "ef_search"}


def test_ui_offers_performance_cases_only():
    # Capacity and streaming cases need an index that is searchable while loading.
    assert set(CASE_CONFIG_MAP[DB.RaBitQLib]) == {CaseLabel.Performance}


def test_ui_defaults_match_the_config_defaults():
    config = ui_task({})
    defaults = RaBitQLibHNSWConfig(metric_type=MetricType.COSINE)

    assert isinstance(config, RaBitQLibHNSWConfig)
    assert config.index_param() == defaults.index_param()
    assert config.search_param() == defaults.search_param()


def test_tuned_values_reach_the_client():
    config = ui_task({"nbits": 5, "nlist": 64, "M": 32, "ef_construction": 400, "ef_search": 256})

    assert config.index_param() == {
        "metric": "ip",
        "nbits": 5,
        "num_clusters": 64,
        "M": 32,
        "ef_construction": 400,
        "seed": 100,
    }
    assert config.search_param() == {"ef": 256}


def test_input_ranges_match_the_cli():
    inputs = {i.displayLabel: i.inputConfig for i in get_case_config_inputs(DB.RaBitQLib, CaseLabel.Performance)}

    assert (inputs["nbits"]["min"], inputs["nbits"]["max"]) == (1, 9)
    assert inputs["num_clusters"]["min"] == 1
    assert inputs["M"]["min"] == 2
    assert inputs["ef_construction"]["min"] == 1
    assert inputs["ef_search"]["min"] == 1
