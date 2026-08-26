# LAION-100M Large-TopK

VDBBench now supports K values through 1,000,000 on the existing `Performance768D100M` case.

## Dataset Selection

The case selects hosted LAION artifacts from K:

- K up to 1,000 uses `test.parquet` and `neighbors.parquet`.
- K from 1,001 through 100,000 uses the 200-query `test_nq200.parquet` and `neighbors_top100k_nq200.parquet` files.
- K from 100,001 through 1,000,000 uses `test_nq200.parquet` and `neighbors_top1m_nq200.parquet`.

Integer-filter runs use their original 1,000-query artifacts through K=1,000. Above that, VDBBench uses the smallest published 200-query GT tier that covers K:

| Integer filter rate | Published GT widths | Maximum K |
|---:|---:|---:|
| 50%, 60%, 70%, 80%, 90%, 95%, 98%, 99% | 100K, 1M | 1M |
| 99.5% | 100K, 500K | 500K |
| 99.8% | 100K, 200K | 200K |
| 99.9% | 100K | 100K |

The loader verifies query ID alignment, row count, and ground-truth width. Unsupported integer rates, requests above the selected filter's maximum K, label-filter runs above K=1,000, and unfiltered LAION K above 1,000,000 fail before database initialization. LAION-backed workloads that do not measure recall, such as cold latency, keep the standard 1,000-query artifacts while forwarding their configured K to the backend.

## Memory And Metrics

Ground truth is represented by a local Parquet path and compact metadata in the parent process. The serial-search subprocess opens that path and reads one Arrow/NumPy neighbor row at a time, avoiding conversion of the full wide GT into Python integer lists.

Recall and NDCG now use O(K) hash lookups. Recall counts each matching returned ID at most once, so duplicate IDs do not inflate the score. A large-TopK serial run reports:

- primary recall at the requested K;
- `recall_at` for each supported cutoff no greater than K;
- serial p50, p95, and p99 latency.

Serial and concurrent latency fields are stored in seconds, matching the existing p95/p99 fields; the frontend converts them to milliseconds for display. `recall_at` values are ratios from 0 to 1. Existing result files load with zero/empty defaults for the new fields.

## Milvus Collection Mode

For Milvus and Zilliz Cloud performance runs with K above 16,384, VDBBench sets `query_mode=large_topk` when it creates the collection, before creating the vector index. The run log records the target database, the requested K, and the selected query mode. When reusing a collection, VDBBench validates the property and fails before loading or searching if the collection is incompatible.

Self-hosted Milvus requires 2.6.14 or later, the release that introduced the `query_mode=large_topk` collection property. Earlier servers accept the property without honoring it, so the requested K still fails against the default 16,384 limit, and the reuse check rejects the collection on the next run.

Other backends are unchanged. Their target collection must already support the requested result count before the benchmark starts.
