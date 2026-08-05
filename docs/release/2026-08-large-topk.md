# LAION-100M Large-TopK

VDBBench now supports K values through 1,000,000 on the existing `Performance768D100M` case.

## Dataset Selection

The case selects hosted LAION artifacts from K:

- K up to 1,000 uses `test.parquet` and `neighbors.parquet`.
- K from 1,001 through 100,000 uses the 200-query `test_nq200.parquet` and `neighbors_top100k_nq200.parquet` files.
- K from 100,001 through 1,000,000 uses `test_nq200.parquet` and `neighbors_top1m_nq200.parquet`.

The loader verifies query ID alignment, row count, and ground-truth width. Filtered LAION performance runs above K=1,000 and LAION performance K values above 1,000,000 fail before database initialization. LAION-backed workloads that do not measure recall, such as cold latency, keep the standard 1,000-query artifacts while forwarding their configured K to the backend.

## Memory And Metrics

Ground truth is represented by a local Parquet path and compact metadata in the parent process. The serial-search subprocess opens that path and reads one Arrow/NumPy neighbor row at a time, avoiding conversion of the full wide GT into Python integer lists.

Recall and NDCG now use O(K) hash lookups. A large-TopK serial run reports:

- primary recall at the requested K;
- `recall_at` for each supported cutoff no greater than K;
- serial p50, p95, and p99 latency.

Serial and concurrent latency fields are stored in seconds, matching the existing p95/p99 fields; the frontend converts them to milliseconds for display. `recall_at` values are ratios from 0 to 1. Existing result files load with zero/empty defaults for the new fields.

## Zilliz Cloud Collection Mode

For Zilliz Cloud performance runs with K above 16,384, VDBBench sets `query_mode=large_topk` when it creates the collection, before creating the vector index. The run log records the requested K and selected query mode. When reusing a collection, VDBBench validates the property and fails before loading or searching if the collection is incompatible.

Milvus and other backends are unchanged. Their target collection must already support the requested result count before the benchmark starts.
