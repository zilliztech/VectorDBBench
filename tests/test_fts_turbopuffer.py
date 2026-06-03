import inspect
import pickle
from types import SimpleNamespace

import pytest
from turbopuffer.resources.namespaces import NamespacesResource

from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferFtsConfig
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer
from vectordb_bench.backend.payload import PayloadProfile


def make_fts_db():
    db = TurboPuffer.__new__(TurboPuffer)
    db._is_fts = True
    db._scalar_id_field = "id"
    db._text_field = "text"
    return db


def make_vector_db(with_scalar_labels=False):
    db = TurboPuffer.__new__(TurboPuffer)
    db.ns = None
    db._scalar_id_field = "id"
    db._vector_field = "vector"
    db._scalar_label_field = "label"
    db._scalar_payload_label_field = "label"
    db.metric = "cosine_distance"
    db.with_scalar_labels = with_scalar_labels
    db.db_case_config = SimpleNamespace(disable_backpressure=False)
    return db


def test_turbopuffer_fts_config_defaults():
    config = TurboPufferFtsConfig()

    assert config.index_param() == {}
    assert config.search_param() == {}


def test_turbopuffer_declares_full_text_support():
    assert TurboPuffer.supports_full_text_search() is True


def test_turbopuffer_recreates_sdk_client_after_pickle(monkeypatch):
    created = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(kwargs)

    monkeypatch.setattr("vectordb_bench.backend.clients.turbopuffer.turbopuffer.tpuf.Turbopuffer", Client)

    db = TurboPuffer(
        dim=0,
        db_config={
            "api_key": "key",
            "region": "aws-us-west-2",
            "api_base_url": "https://api.turbopuffer.com",
            "namespace": "namespace",
        },
        db_case_config=TurboPufferFtsConfig(),
    )
    db.ns = object()
    db._ns_cache = {"namespace": object()}

    restored = pickle.loads(pickle.dumps(db))

    assert restored.ns is None
    assert restored._ns_cache == {}
    assert isinstance(restored.client, Client)
    assert created == [
        {"api_key": "key", "region": "aws-us-west-2", "base_url": "https://api.turbopuffer.com"},
    ]


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


def test_turbopuffer_search_documents_requests_text_payload():
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

    assert db.search_documents("hello", k=7, payload_profile=PayloadProfile.TEXT) == ["d1"]
    assert calls["include_attributes"] == ["text"]


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


def test_turbopuffer_insert_embeddings_without_labels_uses_upsert_columns():
    db = make_vector_db()
    calls = {}

    class Namespace:
        def write(self, **kwargs):
            calls.update(kwargs)

    db.ns = Namespace()

    assert db.insert_embeddings([[0.1, 0.2]], [1]) == (1, None)
    assert calls == {
        "upsert_columns": {
            "id": [1],
            "vector": [[0.1, 0.2]],
        },
        "distance_metric": "cosine_distance",
        "disable_backpressure": False,
    }


def test_turbopuffer_insert_embeddings_with_labels_uses_upsert_columns():
    db = make_vector_db(with_scalar_labels=True)
    calls = {}

    class Namespace:
        def write(self, **kwargs):
            calls.update(kwargs)

    db.ns = Namespace()

    assert db.insert_embeddings([[0.1, 0.2]], [1], labels_data=["a"]) == (1, None)
    assert calls == {
        "upsert_columns": {
            "id": [1],
            "vector": [[0.1, 0.2]],
            "label": ["a"],
        },
        "distance_metric": "cosine_distance",
        "disable_backpressure": False,
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
