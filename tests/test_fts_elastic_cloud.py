from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.elastic_cloud.elastic_cloud import ElasticCloud


def test_elastic_cloud_fts_config_defaults():
    config = ElasticCloudFtsConfig()

    assert config.index_param()["properties"]["text"]["type"] == "text"
    assert config.search_param() == {}


def test_elastic_cloud_declares_full_text_support():
    assert ElasticCloud.supports_full_text_search() is True


def test_elastic_cloud_search_documents_builds_match_query(monkeypatch):
    db = ElasticCloud.__new__(ElasticCloud)
    db.indice = "idx"
    db.id_col_name = "doc_id"
    db.text_col_name = "text"
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
