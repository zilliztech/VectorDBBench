"""Offline unit tests for MongoDBIndexConfig.search_param().

mongodb.py indexes search_params["exact"] on every query. These tests freeze
that contract so a missing key cannot regress into a KeyError.
"""

from vectordb_bench.backend.clients.mongodb.config import MongoDBIndexConfig


def test_search_param_includes_exact_default_false():
    params = MongoDBIndexConfig().search_param()
    assert params["exact"] is False
    assert params["num_candidates_ratio"] == 10


def test_search_param_forwards_exact_true():
    params = MongoDBIndexConfig(exact=True).search_param()
    assert params["exact"] is True
    assert params["num_candidates_ratio"] == 10
