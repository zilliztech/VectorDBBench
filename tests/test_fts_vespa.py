import pytest

from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig
from vectordb_bench.backend.clients.vespa import vespa as vespa_module
from vectordb_bench.backend.clients.vespa.vespa import Vespa
from vectordb_bench.backend.payload import PayloadProfile


def make_fts_db():
    db = Vespa.__new__(Vespa)
    db.schema_name = "docs"
    db._is_fts = True
    db._text_field = "text"
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
    assert calls["default-index"] == "text"


def test_vespa_search_documents_requests_text_payload():
    db = make_fts_db()
    calls = {}

    class Result:
        def get_json(self):
            return {"root": {"children": [{"fields": {"id": "d1", "text": "hello"}}]}}

    class Client:
        def query(self, query):
            calls.update(query)
            return Result()

    db.client = Client()

    assert db.search_documents("hello world", k=4, payload_profile=PayloadProfile.TEXT) == ["d1"]
    assert calls["yql"] == "select id, text from docs where userQuery()"


def test_vespa_search_documents_skips_malformed_hits_and_falls_back_to_document_id():
    db = make_fts_db()

    class Result:
        def get_json(self):
            return {
                "root": {
                    "children": [
                        {"fields": {"id": "from-fields"}},
                        {"id": "id:docs:docs::from-vespa-id", "fields": {"text": "missing id"}},
                        {"fields": {"text": "missing id"}},
                        {"id": "id:docs:docs::only-top-level"},
                        {},
                    ]
                }
            }

    class Client:
        def query(self, query):
            return Result()

    db.client = Client()

    assert db.search_documents("hello") == ["from-fields", "from-vespa-id", "only-top-level"]


@pytest.mark.parametrize("payload", [{"root": {}}, {"root": {"children": None}}])
def test_vespa_search_documents_returns_empty_list_for_missing_children(payload):
    db = make_fts_db()

    class Result:
        def get_json(self):
            return payload

    class Client:
        def query(self, query):
            return Result()

    db.client = Client()

    assert db.search_documents("hello") == []


def test_vespa_insert_documents_feeds_text_documents():
    db = make_fts_db()
    calls = {}

    class Client:
        def feed_iterable(self, data, schema_name, callback=None):
            calls["data"] = list(data)
            calls["schema_name"] = schema_name
            calls["callback"] = callback

    db.client = Client()

    assert db.insert_documents(["alpha", "beta"], ["d1", "d2"]) == (2, None)
    assert calls == {
        "data": [
            {"id": "d1", "fields": {"id": "d1", "text": "alpha"}},
            {"id": "d2", "fields": {"id": "d2", "text": "beta"}},
        ],
        "schema_name": "docs",
        "callback": calls["callback"],
    }
    assert callable(calls["callback"])


def test_vespa_insert_documents_reports_feed_failures(monkeypatch):
    db = make_fts_db()
    warnings = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def is_successful(self):
            return self.status_code == 200

        def get_json(self):
            return {"message": "bad feed"}

    class Client:
        def feed_iterable(self, data, schema_name, callback=None):
            list(data)
            callback(Response(200), "d1")
            callback(Response(500), "d2")

    db.client = Client()
    monkeypatch.setattr(vespa_module.log, "warning", lambda *args, **kwargs: warnings.append(args[0]))

    count, err = db.insert_documents(["alpha", "beta"], ["d1", "d2"])

    assert count == 1
    assert isinstance(err, RuntimeError)
    assert "d2" in str(err)
    assert warnings
    assert warnings[0].startswith("Vespa feed failed")


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
