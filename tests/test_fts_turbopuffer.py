import inspect

import pytest
from turbopuffer.resources.namespaces import NamespacesResource

from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferFtsConfig
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer


def make_fts_db():
    db = TurboPuffer.__new__(TurboPuffer)
    db._is_fts = True
    db._scalar_id_field = "id"
    db._text_field = "text"
    return db


def test_turbopuffer_fts_config_defaults():
    config = TurboPufferFtsConfig()

    assert config.index_param() == {}
    assert config.search_param() == {}


def test_turbopuffer_declares_full_text_support():
    assert TurboPuffer.supports_full_text_search() is True


def test_turbopuffer_sdk_write_uses_upsert_columns_keyword():
    params = inspect.signature(NamespacesResource.write).parameters

    assert "upsert_columns" in params
    assert "columns" not in params


def test_turbopuffer_search_documents_uses_bm25_rank_by():
    db = make_fts_db()
    calls = {}

    class Row:
        def __init__(self, id):
            self.id = id

    class Result:
        rows = [Row("d1")]

    class Namespace:
        def query(self, **kwargs):
            calls.update(kwargs)
            return Result()

    db.ns = Namespace()

    assert db.search_documents("hello", k=7) == ["d1"]
    assert calls["rank_by"] == ("text", "BM25", "hello")
    assert calls["top_k"] == 7


def test_turbopuffer_insert_documents_writes_text_schema():
    db = make_fts_db()
    calls = {}

    class Namespace:
        def write(self, **kwargs):
            calls.update(kwargs)

    db.ns = Namespace()

    assert db.insert_documents(["alpha", "beta"], ["d1", "d2"]) == (2, None)
    assert calls == {
        "upsert_columns": {
            "id": ["d1", "d2"],
            "text": ["alpha", "beta"],
        },
        "schema": {
            "text": {
                "type": "string",
                "full_text_search": True,
            }
        },
    }


def test_turbopuffer_insert_documents_validates_lengths():
    db = make_fts_db()

    class Namespace:
        def write(self, **kwargs):
            raise AssertionError("write should not be called")

    db.ns = Namespace()

    with pytest.raises(ValueError, match="Mismatch between texts .* and doc_ids .* lengths"):
        db.insert_documents(["alpha", "beta"], ["d1"])


def test_turbopuffer_document_methods_guard_non_fts_mode():
    db = TurboPuffer.__new__(TurboPuffer)
    db._is_fts = False
    db.ns = object()

    with pytest.raises(RuntimeError, match="TurboPuffer full-text insert requires TurboPufferFtsConfig"):
        db.insert_documents(["alpha"], ["d1"])

    with pytest.raises(RuntimeError, match="TurboPuffer full-text search requires TurboPufferFtsConfig"):
        db.search_documents("alpha")


@pytest.mark.parametrize("rows", [None, []])
def test_turbopuffer_search_documents_returns_empty_list_for_missing_rows(rows):
    db = make_fts_db()

    class Result:
        pass

    class Namespace:
        def query(self, **kwargs):
            result = Result()
            result.rows = rows
            return result

    db.ns = Namespace()

    assert db.search_documents("hello") == []


def test_turbopuffer_search_documents_returns_empty_list_when_rows_attr_missing():
    db = make_fts_db()

    class Result:
        pass

    class Namespace:
        def query(self, **kwargs):
            return Result()

    db.ns = Namespace()

    assert db.search_documents("hello") == []
