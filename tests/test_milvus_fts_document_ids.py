from vectordb_bench.backend.clients.milvus import milvus as milvus_module
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig


def test_milvus_fts_primary_field_uses_varchar(monkeypatch):
    fields = []

    def fake_field_schema(*args, **kwargs):
        field = {
            "name": kwargs.get("name", args[0] if args else None),
            "dtype": kwargs.get("dtype", args[1] if len(args) > 1 else None),
            "max_length": kwargs.get("max_length"),
            "is_primary": kwargs.get("is_primary", False),
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

    milvus_module.Milvus(
        dim=0,
        db_config={},
        db_case_config=MilvusFtsConfig(),
        collection_name="test_fts",
    )

    primary_field = next(field for field in fields if field["name"] == "doc_id")
    assert primary_field == {
        "name": "doc_id",
        "dtype": milvus_module.DataType.VARCHAR,
        "max_length": 512,
        "is_primary": True,
    }


def test_milvus_fts_insert_documents_stores_string_ids():
    rows = []

    class FakeInsertResult:
        primary_keys = ["123"]

    class FakeCollection:
        def insert(self, inserted_rows):
            rows.extend(inserted_rows)
            return FakeInsertResult()

    db = object.__new__(milvus_module.Milvus)
    db.name = "Milvus"
    db._is_fts = True
    db.col = FakeCollection()
    db.batch_size = 1000
    db._primary_field = "doc_id"
    db._text_field = "text"
    db.with_scalar_labels = False

    insert_count, error = db.insert_documents(texts=["hello"], doc_ids=[123])

    assert insert_count == 1
    assert error is None
    assert rows == [{"doc_id": "123", "text": "hello"}]


def test_milvus_fts_search_documents_returns_string_ids():
    class FakeEntity:
        def get(self, field_name):
            assert field_name == "doc_id"
            return 123

    class FakeHit:
        entity = FakeEntity()

    class FakeCollection:
        def search(self, **kwargs):
            return [[FakeHit()]]

    class FakeCaseConfig:
        def search_param(self):
            return {}

    db = object.__new__(milvus_module.Milvus)
    db._is_fts = True
    db.col = FakeCollection()
    db._primary_field = "doc_id"
    db._sparse_field = "sparse_vector"
    db.case_config = FakeCaseConfig()

    assert db.search_documents(query="hello", k=10) == ["123"]
