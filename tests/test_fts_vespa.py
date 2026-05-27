import pytest

from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig
from vectordb_bench.backend.clients.vespa.vespa import Vespa


def make_fts_db():
    db = Vespa.__new__(Vespa)
    db.schema_name = "docs"
    db._is_fts = True
    db.client = object()
    return db


def test_vespa_fts_config_defaults():
    config = VespaFtsConfig()

    assert config.index_param() == {}
    assert config.search_param() == {}


def test_vespa_declares_full_text_support():
    assert Vespa.supports_full_text_search() is True


def test_vespa_search_documents_uses_user_query():
    db = make_fts_db()
    calls = {}

    class Result:
        def get_json(self):
            return {"root": {"children": [{"fields": {"id": "d1"}}]}}

    class Client:
        def query(self, query):
            calls.update(query)
            return Result()

    db.client = Client()

    assert db.search_documents("hello world", k=4) == ["d1"]
    assert "userQuery()" in calls["yql"]
    assert calls["query"] == "hello world"
    assert calls["ranking"] == "bm25"
    assert calls["hits"] == 4


def test_vespa_insert_documents_feeds_text_documents():
    db = make_fts_db()
    calls = {}

    class Client:
        def feed_iterable(self, data, schema_name):
            calls["data"] = list(data)
            calls["schema_name"] = schema_name

    db.client = Client()

    assert db.insert_documents(["alpha", "beta"], ["d1", "d2"]) == (2, None)
    assert calls == {
        "data": [
            {"id": "d1", "fields": {"id": "d1", "text": "alpha"}},
            {"id": "d2", "fields": {"id": "d2", "text": "beta"}},
        ],
        "schema_name": "docs",
    }


def test_vespa_insert_documents_validates_lengths():
    db = make_fts_db()

    class Client:
        def feed_iterable(self, data, schema_name):
            raise AssertionError("feed_iterable should not be called")

    db.client = Client()

    with pytest.raises(ValueError, match="Mismatch between texts .* and doc_ids .* lengths"):
        db.insert_documents(["alpha", "beta"], ["d1"])


def test_vespa_document_methods_guard_non_fts_mode():
    db = Vespa.__new__(Vespa)
    db._is_fts = False
    db.client = object()

    with pytest.raises(RuntimeError, match="Vespa full-text insert requires VespaFtsConfig"):
        db.insert_documents(["alpha"], ["d1"])

    with pytest.raises(RuntimeError, match="Vespa full-text search requires VespaFtsConfig"):
        db.search_documents("alpha")
