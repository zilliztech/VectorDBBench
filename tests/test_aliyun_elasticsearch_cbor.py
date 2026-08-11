import base64
import struct
from types import SimpleNamespace

import cbor2
import elasticsearch
import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.aliyun_elasticsearch.aliyun_elasticsearch import AliyunElasticsearch
from vectordb_bench.backend.clients.aliyun_elasticsearch.config import (
    AliyunElasticsearchIndexConfig,
    AliyunESQueryWireFormat,
)
from vectordb_bench.backend.clients.elastic_cloud.config import ESElementType


class _FakeClient:
    def __init__(self):
        self.search_kwargs = None
        self.request_args = None
        self.transport = self

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return {"hits": {"hits": [{"fields": {"id": [7]}}]}}

    def perform_request(self, *args, **kwargs):
        self.request_args = (args, kwargs)
        return SimpleNamespace(body={"hits": {"hits": [{"fields": {"id": [7]}}]}})


def _client(wire_format: AliyunESQueryWireFormat) -> AliyunElasticsearch:
    client = object.__new__(AliyunElasticsearch)
    client.query_wire_format = wire_format
    client.client = _FakeClient()
    client.db_config = {"hosts": ["http://localhost:9200"]}
    client.case_config = SimpleNamespace(
        element_type=ESElementType.float,
        num_candidates=90,
        use_rescore=True,
        oversample_ratio=4.6,
    )
    client.indice = "cohere10m"
    client.id_col_name = "id"
    client.vector_col_name = "vector"
    client.filter = []
    client.routing_key = None
    return client


def test_aliyun_index_config_defaults_to_base64_and_accepts_cbor():
    default_config = AliyunElasticsearchIndexConfig()
    cbor_config = AliyunElasticsearchIndexConfig(query_wire_format="cbor-f32le")

    assert default_config.query_wire_format == AliyunESQueryWireFormat.base64_f32be
    assert cbor_config.query_wire_format == AliyunESQueryWireFormat.cbor_f32le
    assert DB.AliyunElasticsearch.case_config_cls() is AliyunElasticsearchIndexConfig


def test_base64_query_transport_remains_big_endian_json():
    client = _client(AliyunESQueryWireFormat.base64_f32be)

    assert client.search_embedding([1.0, -2.5, 3.25], k=10) == [7]

    request = client.client.search_kwargs
    encoded = request["knn"]["query_vector"]
    assert base64.b64decode(encoded) == struct.pack(">3f", 1.0, -2.5, 3.25)
    assert request["knn"]["num_candidates"] == 90
    assert request["knn"]["rescore_vector"] == {"oversample": 4.6}


def test_cbor_query_transport_uses_raw_little_endian_float32():
    client = _client(AliyunESQueryWireFormat.cbor_f32le)

    assert client.search_embedding([1.0, -2.5, 3.25], k=10) == [7]

    args, request = client.client.request_args
    assert args == (
        "POST",
        "/cohere10m/_search?filter_path=hits.hits.fields.id",
    )
    assert request["headers"] == {
        "accept": "application/json",
        "content-type": "application/cbor",
    }
    body = cbor2.loads(request["body"])
    assert body["knn"]["query_vector"] == struct.pack("<3f", 1.0, -2.5, 3.25)
    assert body["knn"]["num_candidates"] == 90
    assert body["knn"]["rescore_vector"] == {"oversample": 4.6}
    assert body["docvalue_fields"] == ["id"]
    assert body["stored_fields"] == "_none_"


def test_cbor_init_registers_application_cbor_serializer(monkeypatch: pytest.MonkeyPatch):
    created = {}

    class FakeElasticsearch:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(elasticsearch, "Elasticsearch", FakeElasticsearch)
    client = _client(AliyunESQueryWireFormat.cbor_f32le)

    with client.init():
        serializer = created["serializers"]["application/cbor"]
        payload = {"query_vector": b"\x00\x01"}
        assert cbor2.loads(serializer.dumps(payload)) == payload

    assert not hasattr(client, "client")


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("routing_key", "tenant-a", "does not support routed queries"),
        ("filter", {"term": {"label": "x"}}, "only supports non-filtered vector queries"),
    ],
)
def test_cbor_query_transport_rejects_unsupported_query_shapes(attribute: str, value: object, message: str):
    client = _client(AliyunESQueryWireFormat.cbor_f32le)
    setattr(client, attribute, value)

    with pytest.raises(ValueError, match=message):
        client.search_embedding([1.0], k=1)


def test_cbor_query_transport_rejects_non_float_vectors():
    client = _client(AliyunESQueryWireFormat.cbor_f32le)
    client.case_config.element_type = ESElementType.byte

    with pytest.raises(ValueError, match="element_type=float"):
        client.search_embedding([1.0], k=1)
