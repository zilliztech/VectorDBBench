from vectordb_bench.backend.clients.milvus import milvus as milvus_module
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus


def test_milvus_fts_config_uses_analyzer_max_token_length():
    config = MilvusFtsConfig(
        analyzer_tokenizer="standard",
        analyzer_enable_lowercase=True,
        analyzer_max_token_length=12,
        analyzer_stop_words="the,and",
    )

    params = config.index_param()["analyzer_params"]

    assert params["tokenizer"] == "standard"
    assert "type" not in params
    assert "lowercase" in params["filter"]
    assert {"type": "length", "max": 12} in params["filter"]
    assert {"type": "stop", "stop_words": ["the", "and"]} in params["filter"]


def test_milvus_declares_full_text_support():
    assert Milvus.supports_full_text_search() is True


def test_milvus_fts_text_field_receives_configured_analyzer_params(monkeypatch):
    fields = []

    def fake_field_schema(*args, **kwargs):
        field = {
            "name": kwargs.get("name", args[0] if args else None),
            "analyzer_params": kwargs.get("analyzer_params"),
        }
        fields.append(field)
        return field

    class FakeCollection:
        def __init__(self, *args, **kwargs):
            pass

        def create_index(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            pass

    monkeypatch.setattr(milvus_module, "FieldSchema", fake_field_schema)
    monkeypatch.setattr(milvus_module, "CollectionSchema", lambda *args, **kwargs: {"args": args, "kwargs": kwargs})
    monkeypatch.setattr(milvus_module, "Function", lambda *args, **kwargs: {"args": args, "kwargs": kwargs})
    monkeypatch.setattr(milvus_module, "Collection", FakeCollection)
    monkeypatch.setattr(milvus_module.utility, "has_collection", lambda collection_name: False)

    import pymilvus

    monkeypatch.setattr(pymilvus.connections, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(pymilvus.connections, "disconnect", lambda *args, **kwargs: None)

    config = MilvusFtsConfig(
        analyzer_tokenizer="standard",
        analyzer_enable_lowercase=True,
        analyzer_max_token_length=12,
        analyzer_stop_words="the,and",
    )

    milvus_module.Milvus(
        dim=0,
        db_config={},
        db_case_config=config,
        collection_name="test_fts",
    )

    text_field = next(field for field in fields if field["name"] == "text")
    assert text_field["analyzer_params"] == config.analyzer_param()


def test_milvus_fts_sparse_index_params_exclude_analyzer_params(monkeypatch):
    create_index_calls = []

    class FakeCollection:
        def __init__(self, *args, **kwargs):
            pass

        def create_index(self, *args, **kwargs):
            create_index_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(milvus_module, "Collection", FakeCollection)

    config = MilvusFtsConfig()
    db = object.__new__(milvus_module.Milvus)
    db._is_fts = True
    db.collection_name = "test_fts"
    db.case_config = config
    db._sparse_field = "sparse_vector"
    db._main_index_name = "sparse_vector_idx"
    db._sort_index_field = "doc_id"
    db._sort_index_name = "doc_id_sort_idx"
    db.with_scalar_labels = False

    db.create_index()

    sparse_index_params = create_index_calls[0]["kwargs"]["index_params"]
    assert sparse_index_params == config.sparse_index_param()
    assert "analyzer_params" not in sparse_index_params
