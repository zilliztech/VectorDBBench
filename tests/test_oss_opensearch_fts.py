import sys
import types

import pytest


class _FakeOpenSearch:
    pass


sys.modules.setdefault("opensearchpy", types.SimpleNamespace(OpenSearch=_FakeOpenSearch))

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.oss_opensearch.config import OSSOpenSearchFtsConfig
from vectordb_bench.backend.clients.oss_opensearch.oss_opensearch import OSSOpenSearch
from vectordb_bench.backend.payload import PayloadProfile


def make_fts_db():
    db = OSSOpenSearch.__new__(OSSOpenSearch)
    db.index_name = "idx"
    db.id_col_name = "doc_id"
    db.text_col_name = "text"
    db._is_fts = True
    db.client = object()
    return db


def test_oss_opensearch_fts_config_defaults():
    config = OSSOpenSearchFtsConfig()

    assert config.index_param()["properties"]["doc_id"] == {"type": "keyword"}
    assert config.index_param()["properties"]["text"] == {"type": "text"}
    assert config.search_param() == {}


def test_oss_opensearch_fts_config_supports_bm25_similarity():
    config = OSSOpenSearchFtsConfig(bm25_k1=1.2, bm25_b=0.75)

    assert config.index_param()["properties"]["text"]["similarity"] == "vdbbench_bm25"
    assert config.similarity_settings() == {
        "similarity": {
            "vdbbench_bm25": {
                "type": "BM25",
                "k1": 1.2,
                "b": 0.75,
            }
        }
    }


def test_oss_opensearch_declares_full_text_support():
    assert OSSOpenSearch.supports_full_text_search() is True
    assert DB.OSSOpenSearch.case_config_cls(IndexType.FTS) is OSSOpenSearchFtsConfig


def test_oss_opensearch_create_index_fts_uses_text_mappings_and_settings():
    db = OSSOpenSearch.__new__(OSSOpenSearch)
    db._is_fts = True
    db.case_config = OSSOpenSearchFtsConfig(
        number_of_shards=2,
        number_of_replicas=1,
        refresh_interval="10s",
    )
    db.index_name = "idx"
    calls = {}

    class Indices:
        def create(self, **kwargs):
            calls.update(kwargs)

    class Client:
        indices = Indices()

    db._create_index(Client())

    assert calls == {
        "index": "idx",
        "body": {
            "settings": {
                "index": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "refresh_interval": "10s",
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                }
            },
        },
    }


def test_oss_opensearch_insert_documents_builds_bulk_body():
    db = make_fts_db()
    captured = {}

    class Client:
        def bulk(self, **kwargs):
            captured.update(kwargs)
            return {"errors": False}

    db.client = Client()

    assert db.insert_documents(["alpha", "beta"], ["d1", "d2"]) == (2, None)
    assert captured["body"] == [
        {"index": {"_index": "idx", "_id": "d1"}},
        {"doc_id": "d1", "text": "alpha"},
        {"index": {"_index": "idx", "_id": "d2"}},
        {"doc_id": "d2", "text": "beta"},
    ]


def test_oss_opensearch_insert_documents_reports_partial_bulk_failure():
    db = make_fts_db()

    class Client:
        def bulk(self, **kwargs):
            return {
                "errors": True,
                "items": [
                    {"index": {"_id": "d1", "status": 201}},
                    {
                        "index": {
                            "_id": "d2",
                            "status": 429,
                            "error": {
                                "type": "rejected_execution_exception",
                                "reason": "indexing queue is full",
                            },
                        }
                    },
                ],
            }

    db.client = Client()

    insert_count, error = db.insert_documents(["alpha", "beta"], ["d1", "d2"])

    assert insert_count == 1
    assert isinstance(error, RuntimeError)
    assert "failed for 1/2 documents" in str(error)
    assert "successful=1" in str(error)
    assert "id=d2" in str(error)
    assert "rejected_execution_exception: indexing queue is full" in str(error)


def test_oss_opensearch_insert_documents_rejects_malformed_bulk_error_response():
    db = make_fts_db()

    class Client:
        def bulk(self, **kwargs):
            return {"errors": True}

    db.client = Client()

    insert_count, error = db.insert_documents(["alpha"], ["d1"])

    assert insert_count == 0
    assert isinstance(error, RuntimeError)
    assert str(error) == "OpenSearch FTS bulk response reported errors without an items list"


def test_oss_opensearch_insert_documents_validates_lengths():
    db = make_fts_db()

    class Client:
        def bulk(self, **kwargs):
            raise AssertionError("bulk should not be called")

    db.client = Client()

    with pytest.raises(ValueError, match="Mismatch between texts .* and doc_ids .* lengths"):
        db.insert_documents(["alpha", "beta"], ["d1"])


def test_oss_opensearch_search_documents_builds_match_query():
    db = make_fts_db()
    calls = {}

    class Client:
        def search(self, **kwargs):
            calls.update(kwargs)
            return {"hits": {"hits": [{"fields": {"doc_id": ["d1"]}}]}}

    db.client = Client()

    assert db.search_documents("hello world", k=3) == ["d1"]
    assert calls["index"] == "idx"
    assert calls["body"] == {"query": {"match": {"text": "hello world"}}}
    assert calls["size"] == 3
    assert calls["stored_fields"] == "_none_"
    assert calls["filter_path"] == ["hits.hits._id", "hits.hits.fields.doc_id"]


def test_oss_opensearch_search_documents_requests_text_payload():
    db = make_fts_db()
    calls = {}

    class Client:
        def search(self, **kwargs):
            calls.update(kwargs)
            return {"hits": {"hits": [{"_id": "d1", "_source": {"text": "hello"}}]}}

    db.client = Client()

    assert db.search_documents("hello world", k=3, payload_profile=PayloadProfile.TEXT) == ["d1"]
    assert calls["_source"] == ["text"]
    assert "stored_fields" not in calls
    assert calls["filter_path"] == [
        "hits.hits._id",
        "hits.hits.fields.doc_id",
        "hits.hits._source.text",
    ]


def test_oss_opensearch_document_methods_guard_non_fts_mode():
    db = OSSOpenSearch.__new__(OSSOpenSearch)
    db._is_fts = False
    db.client = object()

    with pytest.raises(RuntimeError, match="OSSOpenSearch full-text insert requires OSSOpenSearchFtsConfig"):
        db.insert_documents(["alpha"], ["d1"])

    with pytest.raises(RuntimeError, match="OSSOpenSearch full-text search requires OSSOpenSearchFtsConfig"):
        db.search_documents("alpha")
