from vectordb_bench.backend.clients.milvus import milvus as milvus_module
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus


class FakeSchema:
    def __init__(self, fields):
        self.fields = fields

    def add_field(self, name, dtype, **kwargs):
        field = {"name": name, "dtype": dtype, **kwargs}
        self.fields.append(field)
        return field

    def add_function(self, function):
        self.function = function


class FakeIndexParams:
    def __init__(self):
        self.calls = []

    def add_index(self, **kwargs):
        self.calls.append(kwargs)


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
    index_params = FakeIndexParams()

    class FakeMilvusClient:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def create_schema():
            return FakeSchema(fields)

        @staticmethod
        def prepare_index_params():
            return index_params

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
    index_params = FakeIndexParams()
    monkeypatch.setattr(
        milvus_module.MilvusClient,
        "prepare_index_params",
        staticmethod(lambda: index_params),
    )

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

    db._build_index_params()

    sparse_index_params = index_params.calls[0]
    expected = config.sparse_index_param()
    assert sparse_index_params == {
        "field_name": "sparse_vector",
        "index_name": "sparse_vector_idx",
        "index_type": expected["index_type"],
        "metric_type": expected["metric_type"],
        "params": expected["params"],
    }
    assert "analyzer_params" not in sparse_index_params["params"]
