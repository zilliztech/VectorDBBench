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

## CLI: search an existing index

Create and populate the target index before running the following command. In particular, a `native_hnsw` index must
be created through the Elasticsearch API; this search-only command leaves its mapping and shard topology unchanged.

```bash
export VDBBENCH_ES_PASSWORD='<Elasticsearch password>'
export VDBBENCH_ES_QUERY_WIRE_FORMAT='cbor-f32le'

vectordbbench aliyunelasticsearch \
  --scheme http --host '<Elasticsearch hostname>' --port 9200 --user elastic \
  --index-name cohere10m_native_hnsw \
  --case-type Performance768D10M \
  --skip-drop-old --skip-load \
  --m 32 --ef-construction 400 \
  --skip-search-serial --search-concurrent --concurrency-duration 220 \
  --k 10 --num-candidates 90 --use-rescore --oversample-ratio 4.6 \
  --num-concurrency 212,216 --db-label cohere10m-top10-dual16-c212-c216
```

`--password` can be supplied explicitly instead of `VDBBENCH_ES_PASSWORD`. CBOR requests carry Basic authentication
even though they use the low-level transport to preserve `application/cbor`.

`--query-wire-format cbor-f32le` also selects CBOR. The environment precedence remains
`VDBBENCH_ALIYUN_ES_QUERY_WIRE_FORMAT` > `VDBBENCH_ES_QUERY_WIRE_FORMAT` > CLI/case configuration.
Without a selection, the default remains `base64-f32be`.

`--index-name` is stored in `AliyunElasticsearchConfig.index_name` and passed to the adapter. Existing index overrides
retain their precedence: `VDBBENCH_ALIYUN_ES_INDEX` > `VDBBENCH_ES_INDEX` > `index_name` > the adapter's `indice` argument.

For Top100, use `--k 100 --num-candidates 408 --oversample-ratio 4.02 --num-concurrency 60,64`.
The command reports normal VectorDBBench full-stage QPS. It does not discard the first 120 seconds or calculate
server-side tail-window QPS; those measurements must be collected and labelled separately.
