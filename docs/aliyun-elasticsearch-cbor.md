# Aliyun Elasticsearch CBOR query transport

The Aliyun Elasticsearch backend supports two query-vector wire formats:

| `query_wire_format` | HTTP payload |
|---|---|
| `base64-f32be` | JSON request with a Base64-encoded big-endian float32 query vector (default) |
| `cbor-f32le` | `application/cbor` request with the query vector encoded as raw little-endian float32 bytes |

Install the Elasticsearch optional dependencies before using either transport:

```bash
pip install 'vectordb-bench[aliyun_elasticsearch]'
```

The web UI exposes **Query wire format** for Aliyun Elasticsearch performance cases. Programmatic tasks can set
the same option on `AliyunElasticsearchIndexConfig`:

```python
from vectordb_bench.backend.clients.aliyun_elasticsearch.config import (
    AliyunESQueryWireFormat,
    AliyunElasticsearchIndexConfig,
)

case_config = AliyunElasticsearchIndexConfig(
    efConstruction=400,
    M=32,
    num_candidates=90,
    use_rescore=True,
    oversample_ratio=4.6,
    query_wire_format=AliyunESQueryWireFormat.cbor_f32le,
)
```

For existing benchmark scripts, `VDBBENCH_ES_QUERY_WIRE_FORMAT=cbor-f32le` overrides the task setting. The more
specific `VDBBENCH_ALIYUN_ES_QUERY_WIRE_FORMAT` variable takes precedence when both are set.

The CBOR request uses the standard `/<index>/_search` endpoint and returns the  same ID list as the Base64 path. It
is intentionally limited to float vectors and non-filtered, unrouted KNN queries so that the request shape remains
compatible with the Aliyun Elasticsearch raw-f32 parser. Unsupported element types, filters, routing, or wire-format
names fail explicitly instead of silently changing the benchmark path.
