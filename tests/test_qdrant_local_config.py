"""Offline unit tests for the QdrantLocal case config.

The UI builds a case config from whatever inputs `CASE_CONFIG_MAP` declares for
a database. QdrantLocal had no entry there while its config model required `m`
and `ef_construct`, so every run failed validation before it started (#796).
"""

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.qdrant_local.config import QdrantLocalIndexConfig
from vectordb_bench.frontend.config.dbCaseConfigs import get_case_config_inputs


def test_config_builds_without_ui_supplied_index_params():
    config = QdrantLocalIndexConfig()

    assert config.m == 16
    assert config.ef_construct == 200


def test_ui_exposes_the_index_params_the_model_needs():
    labels = {i.label.value for i in get_case_config_inputs(DB.QdrantLocal, CaseLabel.Performance)}

    assert {"m", "ef_construct", "on_disk", "hnsw_ef"} <= labels


def test_ui_defaults_round_trip_into_index_and_search_params():
    inputs = get_case_config_inputs(DB.QdrantLocal, CaseLabel.Performance)
    config = DB.QdrantLocal.case_config_cls()(**{i.label.value: i.inputConfig["value"] for i in inputs})

    assert config.index_param() == {
        "distance": "Cosine",
        "m": 16,
        "ef_construct": 200,
        "on_disk": False,
    }
    # hnsw_ef defaults to 0, which means "let Qdrant use ef_construct"
    assert config.search_param() == {"exact": False}


def test_tuned_values_reach_the_client():
    config = QdrantLocalIndexConfig(m=32, ef_construct=256, hnsw_ef=128, on_disk=True)

    assert config.index_param() == {
        "distance": "Cosine",
        "m": 32,
        "ef_construct": 256,
        "on_disk": True,
    }
    assert config.search_param() == {"exact": False, "hnsw_ef": 128}
