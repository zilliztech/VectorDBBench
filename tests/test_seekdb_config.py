"""Offline unit tests for running SeekDB from the web UI (#689).

SeekDB had no entry in `CASE_CONFIG_MAP`, so the UI sent no index type and
`generate_tasks` asked `case_config_cls` for AUTOINDEX, which SeekDB does not
map. Every run failed with "'NoneType' object is not callable".
"""

from pydantic import SecretStr

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.seekdb.config import SeekDBConfig, SeekDBHNSWConfig
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import get_case_config_inputs
from vectordb_bench.models import CaseConfig


def _ui_defaults(case_label: CaseLabel) -> dict:
    # Mirror dbCaseConfigSetting: walk the inputs in order and keep the displayed ones.
    config = {}
    for config_input in get_case_config_inputs(DB.SeekDB, case_label):
        if config_input.isDisplayed(config):
            if "options" in config_input.inputConfig:
                config[config_input.label] = config_input.inputConfig["options"][0]
            else:
                config[config_input.label] = config_input.inputConfig["value"]
    return config


def _generate(case_type: CaseType, case_label: CaseLabel):
    case = CaseConfig(case_id=case_type)
    db_config = SeekDBConfig(host="127.0.0.1", password=SecretStr(""), database="bench")
    return generate_tasks(
        activedDbList=[DB.SeekDB],
        dbConfigs={DB.SeekDB: db_config},
        activedCaseList=[case],
        allCaseConfigs={DB.SeekDB: {case: _ui_defaults(case_label)}},
        batch_size=100,
    )


def test_ui_exposes_every_field_the_hnsw_model_requires():
    for case_label in (CaseLabel.Load, CaseLabel.Performance):
        labels = {i.label.value for i in get_case_config_inputs(DB.SeekDB, case_label)}

        assert {"IndexType", "m", "ef_construction", "ef_search"} <= labels


def test_performance_case_builds_from_ui_defaults():
    (task,) = _generate(CaseType.Performance768D1M, CaseLabel.Performance)
    config = task.db_case_config

    assert isinstance(config, SeekDBHNSWConfig)
    config.metric_type = MetricType.COSINE
    assert config.index_param() == {
        "metric_type": "cosine",
        "index_type": "HNSW",
        "params": {"m": 16, "ef_construction": 256},
    }
    assert config.search_param() == {
        "metric_type": "cosine_distance",
        "params": {"ef_search": 100},
    }


def test_load_case_builds_from_ui_defaults():
    (task,) = _generate(CaseType.CapacityDim960, CaseLabel.Load)

    assert isinstance(task.db_case_config, SeekDBHNSWConfig)
