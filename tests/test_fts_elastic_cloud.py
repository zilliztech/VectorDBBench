import pytest

from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.elastic_cloud import elastic_cloud as elastic_cloud_module
from vectordb_bench.backend.clients.elastic_cloud.elastic_cloud import ElasticCloud


def make_fts_db():
    db = ElasticCloud.__new__(ElasticCloud)
    db.indice = "idx"
    db.id_col_name = "doc_id"
    db.text_col_name = "text"
    db._is_fts = True
    db.client = object()
    return db


def test_elastic_cloud_fts_config_defaults():
    config = ElasticCloudFtsConfig()

    assert config.index_param()["properties"]["doc_id"] == {"type": "keyword"}
    assert config.index_param()["properties"]["text"]["type"] == "text"
    assert config.search_param() == {}


def test_elastic_cloud_declares_full_text_support():
    assert ElasticCloud.supports_full_text_search() is True


def test_elastic_cloud_search_documents_builds_match_query(monkeypatch):
    db = make_fts_db()
    calls = {}

    class Client:
        def search(self, **kwargs):
            calls.update(kwargs)
            return {"hits": {"hits": [{"fields": {"doc_id": ["d1"]}}]}}

    db.client = Client()

    assert db.search_documents("hello world", k=3) == ["d1"]
    assert calls["index"] == "idx"
    assert calls["query"] == {"match": {"text": "hello world"}}
    assert calls["size"] == 3
    assert calls["filter_path"] == ["hits.hits._id", "hits.hits.fields.doc_id"]


def test_elastic_cloud_insert_documents_builds_bulk_actions(monkeypatch):
    db = make_fts_db()
    captured = {}

    def fake_bulk(client, actions):
        captured["client"] = client
        captured["actions"] = actions
        return 2, []

    monkeypatch.setattr(elastic_cloud_module, "bulk", fake_bulk)

    assert db.insert_documents(["alpha", "beta"], ["d1", "d2"]) == (2, None)
    assert captured["client"] is db.client
    assert captured["actions"] == [
        {
            "_index": "idx",
            "_id": "d1",
            "_source": {"doc_id": "d1", "text": "alpha"},
        },
        {
            "_index": "idx",
            "_id": "d2",
            "_source": {"doc_id": "d2", "text": "beta"},
        },
    ]


def test_elastic_cloud_insert_documents_validates_lengths(monkeypatch):
    db = make_fts_db()

    def fail_bulk(client, actions):
        raise AssertionError("bulk should not be called")

    monkeypatch.setattr(elastic_cloud_module, "bulk", fail_bulk)

    with pytest.raises(ValueError, match="Mismatch between texts .* and doc_ids .* lengths"):
        db.insert_documents(["alpha", "beta"], ["d1"])


def test_elastic_cloud_document_methods_guard_non_fts_mode():
    db = ElasticCloud.__new__(ElasticCloud)
    db._is_fts = False
    db.client = object()

    with pytest.raises(RuntimeError, match="ElasticCloud full-text insert requires ElasticCloudFtsConfig"):
        db.insert_documents(["alpha"], ["d1"])

    with pytest.raises(RuntimeError, match="ElasticCloud full-text search requires ElasticCloudFtsConfig"):
        db.search_documents("alpha")


def test_elastic_cloud_search_documents_prefers_elasticsearch_id():
    db = make_fts_db()

    class Client:
        def search(self, **kwargs):
            return {
                "hits": {
                    "hits": [
                        {"_id": "es-id", "fields": {"doc_id": ["field-id"]}},
                        {"fields": {"doc_id": ["fallback-id"]}},
                    ]
                }
            }

    db.client = Client()

    assert db.search_documents("hello") == ["es-id", "fallback-id"]


def test_elastic_cloud_search_documents_empty_hits():
    db = make_fts_db()

    class Client:
        def search(self, **kwargs):
            return {"hits": {"hits": []}}

    db.client = Client()

    assert db.search_documents("hello") == []


def test_elastic_cloud_create_indice_fts_uses_text_mappings_and_settings():
    db = ElasticCloud.__new__(ElasticCloud)
    db._is_fts = True
    db.case_config = ElasticCloudFtsConfig(
        number_of_shards=2,
        number_of_replicas=1,
        refresh_interval="10s",
    )
    db.indice = "idx"
    calls = {}

    class Indices:
        def create(self, **kwargs):
            calls.update(kwargs)

    class Client:
        indices = Indices()

    db._create_indice(Client())

    assert calls == {
        "index": "idx",
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "text": {"type": "text"},
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "refresh_interval": "10s",
            }
        },
    }
