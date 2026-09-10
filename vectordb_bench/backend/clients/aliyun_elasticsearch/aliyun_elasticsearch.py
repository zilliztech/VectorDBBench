import base64
import os
import struct
import sys
from array import array
from contextlib import contextmanager
from urllib.parse import quote

from ..elastic_cloud.config import ElasticCloudIndexConfig, ESElementType
from ..elastic_cloud.elastic_cloud import ElasticCloud
from .config import AliyunESQueryWireFormat

QUERY_WIRE_FORMAT_ENV = "VDBBENCH_ES_QUERY_WIRE_FORMAT"
ALIYUN_QUERY_WIRE_FORMAT_ENV = "VDBBENCH_ALIYUN_ES_QUERY_WIRE_FORMAT"


class AliyunElasticsearch(ElasticCloud):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ElasticCloudIndexConfig,
        indice: str = "vdb_bench_indice",  # must be lowercase
        id_col_name: str = "id",
        vector_col_name: str = "vector",
        drop_old: bool = False,
        **kwargs,
    ):
        configured_wire_format = getattr(
            db_case_config,
            "query_wire_format",
            AliyunESQueryWireFormat.base64_f32be,
        )
        wire_format = os.environ.get(
            ALIYUN_QUERY_WIRE_FORMAT_ENV,
            os.environ.get(QUERY_WIRE_FORMAT_ENV, configured_wire_format),
        )
        try:
            self.query_wire_format = AliyunESQueryWireFormat(wire_format)
        except ValueError as exc:
            supported = ", ".join(value.value for value in AliyunESQueryWireFormat)
            msg = f"Unsupported Aliyun Elasticsearch query wire format {wire_format!r}; expected one of: {supported}"
            raise ValueError(msg) from exc

        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            indice=os.environ.get("VDBBENCH_ALIYUN_ES_INDEX", os.environ.get("VDBBENCH_ES_INDEX", indice)),
            id_col_name=os.environ.get(
                "VDBBENCH_ALIYUN_ES_ID_FIELD",
                os.environ.get("VDBBENCH_ES_ID_FIELD", id_col_name),
            ),
            vector_col_name=vector_col_name,
            drop_old=drop_old,
            **kwargs,
        )

    @contextmanager
    def init(self):
        if self.query_wire_format != AliyunESQueryWireFormat.cbor_f32le:
            with super().init():
                yield
            return

        import cbor2
        from elastic_transport import Serializer
        from elasticsearch import Elasticsearch

        class CborSerializer(Serializer):
            mimetype = "application/cbor"

            def dumps(self, data: object) -> bytes:
                return data if isinstance(data, bytes) else cbor2.dumps(data)

            def loads(self, data: bytes) -> object:
                return cbor2.loads(data)

        serializers = dict(self.db_config.get("serializers", {}))
        serializers[CborSerializer.mimetype] = CborSerializer()
        client_config = {key: value for key, value in self.db_config.items() if key != "serializers"}
        self.client = Elasticsearch(
            **client_config,
            request_timeout=180,
            serializers=serializers,
        )
        try:
            yield
        finally:
            self.client = None
            del self.client

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        if self.query_wire_format == AliyunESQueryWireFormat.cbor_f32le:
            return self._search_embedding_cbor(query, k)

        encoded_query = base64.b64encode(struct.pack(f">{len(query)}f", *query)).decode("ascii")
        return super().search_embedding(encoded_query, k, **kwargs)

    def _search_embedding_cbor(self, query: list[float], k: int) -> list[int]:
        import cbor2

        assert self.client is not None, "should self.init() first"
        if self.case_config.element_type != ESElementType.float:
            msg = "CBOR raw-f32 query transport requires Elasticsearch element_type=float"
            raise ValueError(msg)
        if self.routing_key is not None:
            msg = "CBOR raw-f32 query transport does not support routed queries"
            raise ValueError(msg)
        if self.filter:
            msg = "CBOR raw-f32 query transport only supports non-filtered vector queries"
            raise ValueError(msg)

        values = array("f", query)
        if values.itemsize != 4:
            msg = "array('f') is not float32 on this Python build"
            raise RuntimeError(msg)
        if sys.byteorder != "little":
            values.byteswap()

        knn = {
            "field": self.vector_col_name,
            "k": k,
            "num_candidates": self.case_config.num_candidates,
            "filter": self.filter,
            "query_vector": values.tobytes(),
        }
        if self.case_config.use_rescore:
            knn["rescore_vector"] = {"oversample": self.case_config.oversample_ratio}

        response = self.client.transport.perform_request(
            "POST",
            f"/{quote(self.indice, safe='')}/_search?filter_path=hits.hits.fields."
            f"{quote(self.id_col_name, safe='')}",
            headers={"accept": "application/json", "content-type": "application/cbor"},
            body=cbor2.dumps(
                {
                    "knn": knn,
                    "size": k,
                    "_source": False,
                    "docvalue_fields": [self.id_col_name],
                    "stored_fields": "_none_",
                }
            ),
            request_timeout=180,
        )
        payload = response.body if hasattr(response, "body") else response
        return [int(hit["fields"][self.id_col_name][0]) for hit in payload["hits"]["hits"]]
