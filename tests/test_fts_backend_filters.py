# ruff: noqa: ANN001, ARG001, E402

import io
import sys
import threading
import types

import pytest

sys.modules.setdefault("turbopuffer", types.SimpleNamespace(Turbopuffer=object))

from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.elastic_cloud.elastic_cloud import ElasticCloud
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer
from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig
from vectordb_bench.backend.clients.vespa.vespa import Vespa
from vectordb_bench.backend.filter import NewIntFilter


def test_milvus_fts_insert_and_search_use_filter_id():
    db = Milvus.__new__(Milvus)
    db._is_fts = True
    db.name = "Milvus"
    db.collection_name = "col"
    db.batch_size = 1000
    db.with_scalar_labels = False
    db._primary_field = "doc_id"
    db._text_field = "text"
    db._filter_id_field = "filter_id"
    db._sparse_field = "sparse"
    db.case_config = MilvusFtsConfig()
    calls = {}

    class Client:
        def insert(self, collection_name, rows):
            calls["insert"] = (collection_name, rows)
            return {"insert_count": len(rows)}

        def search(self, **kwargs):
            calls["search"] = kwargs
            return [[{"entity": {"doc_id": "d1"}}]]

    db.client = Client()

    assert db.insert_documents(["alpha"], ["d1"], filter_ids=[12]) == (1, None)
    db.prepare_filter(NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=10))
    assert db.search_documents("alpha") == ["d1"]

    assert calls["insert"] == ("col", [{"doc_id": "d1", "text": "alpha", "filter_id": 12}])
    assert calls["search"]["filter"] == "filter_id >= 10"


def test_elastic_cloud_fts_insert_and_search_use_filter_id(monkeypatch):
    db = ElasticCloud.__new__(ElasticCloud)
    db._is_fts = True
    db.indice = "idx"
    db.id_col_name = "doc_id"
    db.text_col_name = "text"
    db.filter_id_col_name = "filter_id"
    db.filter = None
    db.client = object()
    bulk_calls = []

    def fake_bulk(client, actions):
        bulk_calls.append(actions)
        return len(actions), []

    monkeypatch.setattr("vectordb_bench.backend.clients.elastic_cloud.elastic_cloud.bulk", fake_bulk)
    assert db.insert_documents(["alpha"], ["d1"], filter_ids=[3]) == (1, None)
    assert bulk_calls == [
        [
            {
                "_index": "idx",
                "_id": "d1",
                "_source": {"doc_id": "d1", "text": "alpha", "filter_id": 3},
            }
        ]
    ]

    search_calls = {}

    class Client:
        def search(self, **kwargs):
            search_calls.update(kwargs)
            return {"hits": {"hits": [{"fields": {"doc_id": ["d1"]}}]}}

    db.client = Client()
    db.prepare_filter(NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=3))
    assert db.search_documents("alpha") == ["d1"]
    assert search_calls["query"] == {
        "bool": {
            "must": {"match": {"text": "alpha"}},
            "filter": {"range": {"filter_id": {"gte": 3}}},
        }
    }


def test_elastic_cloud_fts_mapping_has_filter_id():
    assert ElasticCloudFtsConfig().index_param()["properties"]["filter_id"] == {"type": "long"}


def test_vespa_fts_feed_and_search_use_filter_id():
    db = Vespa.__new__(Vespa)
    db._is_fts = True
    db.schema_name = "schema"
    db._text_field = "text"
    db._filter_id_field = "filter_id"
    db._filter_expr = None
    db.case_config = VespaFtsConfig()
    db._feed_lock = threading.Lock()
    db._feed_written_count = 0
    output = io.StringIO()

    db._ensure_fts_feed_client = lambda: types.SimpleNamespace(
        poll=lambda: None,
        stdin=types.SimpleNamespace(write=output.write, flush=lambda: None),
    )

    db._write_fts_feed_batch(["alpha"], ["d1"], [9])
    assert '"filter_id":9' in output.getvalue()

    calls = {}

    class Result:
        def get_json(self):
            return {"root": {"children": [{"fields": {"id": "d1"}}]}}

    def query(payload):
        calls["payload"] = payload
        return Result()

    db.client = types.SimpleNamespace(query=query)
    db.prepare_filter(NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=9))
    assert db.search_documents("alpha") == ["d1"]
    assert calls["payload"]["yql"] == "select id from schema where userQuery() and filter_id >= 9"


def test_turbopuffer_fts_insert_and_search_use_filter_id():
    db = TurboPuffer.__new__(TurboPuffer)
    db._is_fts = True
    db._text_field = "text"
    db._scalar_id_field = "id"
    db._filter_id_field = "filter_id"
    db.expr = None
    db.db_case_config = types.SimpleNamespace(disable_backpressure=False)
    write_calls = {}
    query_calls = {}

    class Namespace:
        def write(self, **kwargs):
            write_calls.update(kwargs)

        def query(self, **kwargs):
            query_calls.update(kwargs)
            return types.SimpleNamespace(rows=[types.SimpleNamespace(id="d1")])

    db.ns = Namespace()

    assert db.insert_documents(["alpha"], ["d1"], filter_ids=[5]) == (1, None)
    db.prepare_filter(NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=5))
    assert db.search_documents("alpha") == ["d1"]

    assert write_calls["upsert_columns"]["filter_id"] == [5]
    assert query_calls["filters"] == ("filter_id", "Gte", 5)


@pytest.mark.parametrize("db_cls", [Milvus, ElasticCloud, Vespa, TurboPuffer])
def test_fts_filter_rejects_non_filter_id_field(db_cls):
    db = db_cls.__new__(db_cls)
    db._is_fts = True
    db._filter_id_field = "filter_id"
    db.filter_id_col_name = "filter_id"

    with pytest.raises(ValueError, match="filter_id"):
        db.prepare_filter(NewIntFilter(filter_rate=0.5, int_field="id", int_value=1))
