import pytest

from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudConfig
from vectordb_bench.backend.clients.elastic_cloud import elastic_cloud as elastic_cloud_module
from vectordb_bench.backend.clients.elastic_cloud.cli import build_elastic_config
from vectordb_bench.backend.clients.elastic_cloud.elastic_cloud import ElasticCloud
from vectordb_bench.backend.payload import PayloadProfile


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


def test_elastic_cloud_config_supports_cloud_id():
    config = ElasticCloudConfig(cloud_id="cloud:abc", password="secret")

    assert config.to_dict() == {
        "cloud_id": "cloud:abc",
        "basic_auth": ("elastic", "secret"),
    }


def test_elastic_cloud_config_supports_self_hosted_http():
    config = ElasticCloudConfig(host="localhost", port=9200)

    assert config.to_dict() == {
        "hosts": ["http://localhost:9200"],
        "verify_certs": True,
    }


def test_elastic_cloud_config_supports_self_hosted_url_and_auth():
    config = ElasticCloudConfig(
        host="https://es.example.com:9243",
        user_name="elastic",
        password="secret",
        verify_certs=False,
    )

    assert config.to_dict() == {
        "hosts": ["https://es.example.com:9243"],
        "basic_auth": ("elastic", "secret"),
        "verify_certs": False,
    }


def test_elastic_cloud_config_requires_cloud_id_or_host():
    with pytest.raises(ValueError, match="Either cloud_id or host"):
        ElasticCloudConfig().to_dict()


def test_elastic_cloud_config_requires_password_for_cloud_id():
    with pytest.raises(ValueError, match="password is required when cloud_id is set"):
        ElasticCloudConfig(cloud_id="cloud:abc").to_dict()


def test_elastic_cloud_cli_builds_self_hosted_config():
    config = build_elastic_config(
        {
            "db_label": "local",
            "cloud_id": None,
            "host": "localhost",
            "port": 9201,
            "user_name": None,
            "password": None,
            "use_ssl": True,
            "verify_certs": False,
        }
    )

    assert config.db_label == "local"
    assert config.to_dict() == {
        "hosts": ["https://localhost:9201"],
        "verify_certs": False,
    }


def test_elastic_cloud_cli_builds_cloud_config():
    config = build_elastic_config(
        {
            "db_label": "cloud",
            "cloud_id": "cloud:abc",
            "host": None,
            "port": 9200,
            "user_name": "bench",
            "password": "secret",
            "use_ssl": False,
            "verify_certs": True,
        }
    )

    assert config.db_label == "cloud"
    assert config.to_dict() == {
        "cloud_id": "cloud:abc",
        "basic_auth": ("bench", "secret"),
    }


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
    assert calls["stored_fields"] == "_none_"
    assert calls["filter_path"] == ["hits.hits._id", "hits.hits.fields.doc_id"]


def test_elastic_cloud_search_documents_requests_text_payload(monkeypatch):
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
