"""Offline unit tests for MongoDBIndexConfig.search_param() and search_embedding.

mongodb.py indexes search_params["exact"] on every query. These tests freeze
that contract so a missing key cannot regress into a KeyError, and they drive
search_embedding with a mocked collection (no live MongoDB).
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

try:
    import pymongo  # noqa: F401
except ImportError:
    _pymongo = types.ModuleType("pymongo")
    _pymongo.MongoClient = MagicMock
    _ops = types.ModuleType("pymongo.operations")
    _ops.SearchIndexModel = MagicMock
    sys.modules["pymongo"] = _pymongo
    sys.modules["pymongo.operations"] = _ops

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.mongodb.config import MongoDBConfig, MongoDBIndexConfig
from vectordb_bench.backend.clients.mongodb.mongodb import MongoDB
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    MongoDBPerformanceConfig,
    get_case_config_inputs,
)
from vectordb_bench.models import CaseConfig, CaseConfigParamType


def test_search_param_includes_exact_default_false():
    params = MongoDBIndexConfig().search_param()
    assert params["exact"] is False
    assert params["num_candidates_ratio"] == 10


def test_search_param_forwards_exact_true():
    params = MongoDBIndexConfig(exact=True).search_param()
    assert params["exact"] is True
    assert params["num_candidates_ratio"] == 10


def _client(case_config=None):
    db = MongoDB.__new__(MongoDB)
    db.vector_field = "vector"
    db.id_field = "id"
    db.case_config = case_config or MongoDBIndexConfig()
    db.collection = MagicMock()
    db.collection.aggregate.return_value = [{"id": 7}, {"id": 8}]
    return db


def test_search_embedding_default_uses_num_candidates():
    db = _client()
    ids = db.search_embedding([0.1, 0.2], k=100)
    assert ids == [7, 8]
    vector_search = db.collection.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert "exact" not in vector_search
    assert vector_search["numCandidates"] == 1000  # min(10000, k * ratio)
    assert vector_search["limit"] == 100
    assert vector_search["queryVector"] == [0.1, 0.2]


def test_search_embedding_exact_true_omits_num_candidates():
    db = _client(MongoDBIndexConfig(exact=True))
    db.search_embedding([0.1, 0.2], k=10)
    vector_search = db.collection.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert vector_search["exact"] is True
    assert "numCandidates" not in vector_search


def test_search_embedding_missing_exact_key_is_ann():
    """.get("exact") must not KeyError if a case config omits the key."""
    cfg = SimpleNamespace(search_param=lambda: {"num_candidates_ratio": 10})
    db = _client(cfg)
    db.search_embedding([0.1, 0.2], k=50)
    vector_search = db.collection.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert "exact" not in vector_search
    assert vector_search["numCandidates"] == 500


def test_performance_ui_exposes_exact_default_false():
    labels = [c.label for c in MongoDBPerformanceConfig]
    assert CaseConfigParamType.mongodb_exact in labels
    exact = next(c for c in MongoDBPerformanceConfig if c.label == CaseConfigParamType.mongodb_exact)
    assert exact.inputConfig["value"] is False
    perf_inputs = get_case_config_inputs(DB.MongoDB, CaseLabel.Performance)
    assert CaseConfigParamType.mongodb_exact in [c.label for c in perf_inputs]


def _tasks_from_ui(overrides=None):
    case = CaseConfig(case_id=CaseType.Performance1536D50K)
    ui = {
        CaseConfigParamType.mongodb_quantization_type: "none",
        CaseConfigParamType.mongodb_num_candidates_ratio: 10,
        CaseConfigParamType.mongodb_exact: False,
    }
    if overrides:
        ui.update(overrides)
    return generate_tasks(
        [DB.MongoDB],
        {DB.MongoDB: MongoDBConfig()},
        [case],
        {DB.MongoDB: {case: ui}},
    )


def test_generate_tasks_forwards_exact_false_by_default():
    tasks = _tasks_from_ui()
    assert len(tasks) == 1
    assert tasks[0].db_case_config.exact is False
    assert tasks[0].db_case_config.search_param()["exact"] is False


def test_generate_tasks_forwards_exact_true_from_ui():
    tasks = _tasks_from_ui({CaseConfigParamType.mongodb_exact: True})
    assert tasks[0].db_case_config.exact is True
    assert tasks[0].db_case_config.search_param()["exact"] is True
