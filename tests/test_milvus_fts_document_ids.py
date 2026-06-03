from vectordb_bench.backend.clients.milvus import milvus as milvus_module
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.payload import PayloadProfile


def test_milvus_fts_primary_field_uses_varchar(monkeypatch):
    fields = []
    index_calls = []

    class FakeSchema:
        def add_field(self, name, dtype, **kwargs):
            field = {"name": name, "dtype": dtype, **kwargs}
            fields.append(field)
            return field

        def add_function(self, function):
            self.function = function

    class FakeIndexParams:
        def add_index(self, **kwargs):
            index_calls.append(kwargs)

    class FakeMilvusClient:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def create_schema():
            return FakeSchema()

        @staticmethod
        def prepare_index_params():
            return FakeIndexParams()

        def has_collection(self, collection_name):
            return False

        def create_collection(self, *args, **kwargs):
            pass

        def create_index(self, *args, **kwargs):
            pass

        def load_collection(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(milvus_module, "MilvusClient", FakeMilvusClient)

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

    class FakeClient:
        def insert(self, collection_name, inserted_rows):
            assert collection_name == "test_fts"
            rows.extend(inserted_rows)
            return {"insert_count": len(inserted_rows), "primary_keys": FakeInsertResult.primary_keys}

    db = object.__new__(milvus_module.Milvus)
    db.name = "Milvus"
    db._is_fts = True
    db.client = FakeClient()
    db.collection_name = "test_fts"
    db.batch_size = 1000
    db._primary_field = "doc_id"
    db._text_field = "text"
    db.with_scalar_labels = False

    insert_count, error = db.insert_documents(texts=["hello"], doc_ids=[123])

    assert insert_count == 1
    assert error is None
    assert rows == [{"doc_id": "123", "text": "hello"}]


def test_milvus_fts_search_documents_returns_string_ids():
    calls = {}

    class FakeClient:
        def search(self, **kwargs):
            calls.update(kwargs)
            assert kwargs["collection_name"] == "test_fts"
            assert kwargs["anns_field"] == "sparse_vector"
            return [[{"entity": {"doc_id": 123}}]]

    class FakeCaseConfig:
        def search_param(self):
            return {}

    db = object.__new__(milvus_module.Milvus)
    db._is_fts = True
    db.client = FakeClient()
    db.collection_name = "test_fts"
    db._primary_field = "doc_id"
    db._text_field = "text"
    db._sparse_field = "sparse_vector"
    db.case_config = FakeCaseConfig()

    assert db.search_documents(query="hello", k=10) == ["123"]
    assert calls["output_fields"] == ["doc_id"]


def test_milvus_fts_search_documents_requests_text_payload():
    calls = {}

    class FakeClient:
        def search(self, **kwargs):
            calls.update(kwargs)
            return [[{"entity": {"doc_id": 123, "text": "hello"}}]]

    class FakeCaseConfig:
        def search_param(self):
            return {}

    db = object.__new__(milvus_module.Milvus)
    db.name = "Milvus"
    db._is_fts = True
    db.client = FakeClient()
    db.collection_name = "test_fts"
    db._primary_field = "doc_id"
    db._text_field = "text"
    db._sparse_field = "sparse_vector"
    db.case_config = FakeCaseConfig()

    assert db.search_documents(query="hello", k=10, payload_profile=PayloadProfile.TEXT) == ["123"]
    assert calls["output_fields"] == ["doc_id", "text"]
