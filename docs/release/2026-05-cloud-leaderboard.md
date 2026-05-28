# VectorDBBench Cloud Leaderboard Release Note

May 2026

The VectorDBBench Cloud Leaderboard moves beyond a single raw-throughput ranking. It evaluates managed vector databases around the behaviors production teams have to plan for: ingest readiness, payload-aware search, tenant-shaped workloads, cold latency, and cost at practical QPS targets.

## Why we need a new leaderboard now

The vector database market has moved past the "highest QPS wins" phase. Production teams choosing a managed vector database also care about budget, data freshness, tail latency, recall, metadata payloads, tenant isolation, and operational predictability.

The existing VectorDBBench leaderboard remains useful for comparing baseline search performance across systems. But cloud buyers ask a wider set of questions:

- When does newly inserted data become searchable?
- When is it fully indexed?
- What happens when search returns metadata or vectors instead of only IDs?
- What happens when traffic is split across many tenants?
- What does each reachable QPS tier cost?

The Cloud Leaderboard is designed around those questions. It keeps performance visible, but puts it next to the readiness, payload, tenant, cold-start, and cost signals that determine what a customer can safely deploy.

## What the Cloud Leaderboard changes

The Cloud Leaderboard is a production cloud decision layer, not a replacement for the original raw-performance board. The main change is that benchmark cases now model cloud operating concerns directly instead of treating all products as simple warm search engines.

The new cases add:

- Insert readiness measurement, including client insert completion, searchable delay, and indexed delay.
- Explicit response payload profiles: IDs only, scalar label metadata, or vector values.
- Cloud cold-latency measurement for the first search path after idle or cache-cold conditions.
- Multi-tenant search, where data is split into deterministic tenant labels or namespaces and queries are routed by tenant.
- Cost-oriented interpretation, so raw QPS can be read together with monthly cost and readiness constraints.

This matters because a top-line QPS table can hide important tradeoffs. A system can look strong on peak throughput while losing ground on recall, p99 latency, cold-start behavior, payload cost, or sustained cost at the same target QPS.

## Who we tested this round

This round focuses on three popular cloud vector databases:

- Zilliz Cloud, including tiered and fixed-capacity configurations.
- turbopuffer, including normal, pinned, and backpressure-related configurations where applicable.
- Pinecone serverless.

The tested matrix is intentionally cloud-oriented. It compares managed products and managed-service modes rather than only local or self-hosted engine behavior.

## The new tests we added

Version 2 adds four cloud-oriented cases in VectorDBBench. Each case is designed to expose a production behavior that a plain QPS benchmark can miss.

### CloudInsertCase

**Purpose.** CloudInsertCase measures write readiness, not just client-side insert speed. This is important for backfills, migrations, daily refreshes, and release workflows where a team needs to know when newly written vectors can safely take traffic.

**How it works.** The case loads the dataset with `ConcurrentInsertRunner`, records insert completion time and rows per second, then polls the database until inserted data is fully searchable and fully indexed. The resulting metric separates:

- `insert_completion_seconds`
- `insert_rows_per_second`
- `searchable_after_insert_seconds`
- `indexed_after_searchable_seconds`

Example: run LAION 100M insert readiness on Zilliz Cloud with a 10k batch size.

```bash
vectordbbench zillizautoindex \
  --case-type CloudInsertCase \
  --uri "$ZILLIZ_URI" \
  --token "$ZILLIZ_TOKEN" \
  --collection-name cloud_insert_laion100m_bs10k \
  --cloud-insert-batch-size 10000 \
  --load-concurrency 16 \
  --skip-search-serial \
  --skip-search-concurrent \
  --task-label cloud-insert-zilliz-12cu
```

### CloudPayloadSearchCase

**Purpose.** CloudPayloadSearchCase measures search when the response body resembles production traffic. Many applications return more than vector IDs: they return scalar metadata, labels, or the vector values themselves. That response payload can change throughput, latency, and even product ranking.

**How it works.** The case extends the normal performance case with an explicit `payload_profile`. Supported profiles are:

- `ids_only`
- `scalar_label`
- `vector`

The case can also run unfiltered search, integer-filter search through `--cloud-filter-rate`, or scalar-label filter search through `--cloud-label-percentage`. It records QPS, latency, recall where applicable, and estimated response payload bytes per query.

When `payload_profile` is `scalar_label`, VectorDBBench materializes scalar label data even for unfiltered runs. This keeps the loaded schema aligned with the requested response payload instead of only loading labels for scalar-label filter runs.

Example: run vector-payload search on Pinecone with a highly selective integer filter.

```bash
vectordbbench pinecone \
  --case-type CloudPayloadSearchCase \
  --api-key "$PINECONE_API_KEY" \
  --index-name "$PINECONE_INDEX" \
  --payload-profile vector \
  --cloud-filter-rate 0.001 \
  --k 100 \
  --num-concurrency 60,80 \
  --concurrency-duration 30 \
  --task-label cloud-payload-pinecone-vector-int-filter-0-1p
```

### CloudMultiTenantSearchCase

**Purpose.** CloudMultiTenantSearchCase models SaaS-shaped traffic. Instead of treating the dataset as one flat global collection, it splits records across many tenants and routes each query to a tenant. This highlights products whose namespace, partition-key, or tenant-filter paths behave differently from single-tenant search.

**How it works.** The case defaults to the Cohere 10M dataset and assigns each row to a deterministic tenant by `row_id % tenant_count`. During search, queries are routed to the corresponding tenant label or namespace. The case supports the same payload profiles and optional filter modes as payload search.

Tenant routing labels and scalar payload labels are separate concepts. A multi-tenant run can route by tenant while still storing and returning scalar-label payload data when `payload_profile` is `scalar_label`, and scalar-label filters continue to use the scalar label field rather than the tenant routing field.

TurboPuffer tenant namespace cache warmup is explicit. By default, `CloudMultiTenantSearchCase` does not warm tenant namespaces during `optimize()`; use `--multitenant-warmup-policy all` only when the benchmark should model proactively warmed tenant namespaces.

Example: run 1,000-tenant IDs-only search on turbopuffer.

```bash
vectordbbench turbopuffer \
  --case-type CloudMultiTenantSearchCase \
  --dataset-with-size-type "Large Cohere (768dim, 10M)" \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region aws-us-east-1 \
  --namespace vdbbench_mt_seed \
  --multitenant-namespace-prefix vdbbench_mt_ \
  --tenant-count 1000 \
  --tenant-prefix tenant_ \
  --tenant-id-width 4 \
  --payload-profile ids_only \
  --num-concurrency 40,60,80 \
  --concurrency-duration 30 \
  --task-label cloud-multitenant-turbopuffer-1000t
```

### CloudColdLatencyCase

**Purpose.** CloudColdLatencyCase measures first-query and cold-path latency that warm benchmark loops can hide. This matters for serverless products, storage-tiered products, idle workloads, and customer-facing applications where the first query after an idle period is visible to users.

**How it works.** The case is intentionally search-only and must run against an existing collection that has already become cold according to the product's cache and storage behavior. It rejects `drop_old` and `load` stages because insert-then-immediately-search runs can leave caches, indexing paths, or vendor warmup APIs in an ambiguous state. `ColdWarmSearchRunner` runs serial searches in cold and warm passes. It records cold-latency details in `additional_parameters["cold_latency"]` and also records payload profile and estimated payload bytes per query.

Example: run a pinned turbopuffer cold-latency test with scalar-label payloads.

```bash
vectordbbench turbopuffer \
  --case-type CloudColdLatencyCase \
  --skip-drop-old \
  --skip-load \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region aws-us-east-1 \
  --namespace cloud_cold_latency_scalar_label \
  --pin-namespace \
  --pin-replicas 2 \
  --payload-profile scalar_label \
  --cloud-cold-query-count 1000 \
  --skip-search-concurrent \
  --task-label cloud-cold-latency-turbopuffer-pinned-scalar-label
```

## Caveats

This release note introduces the new Cloud Leaderboard direction; it is not the full benchmark report. Detailed tables, raw JSON artifacts, pricing worksheets, and edge-case analysis should live in the benchmark report or external result artifact repository.

Important caveats:

- Pricing changes over time. Cost charts need a pricing date, region, and configuration assumptions.
- Managed-service configuration can materially change results, especially for serverless scaling, pinned replicas, capacity units, and storage-tiering modes.
- "Fully indexed" and "fully searchable" readiness may be exposed differently by each vendor, so the implementation must document how each status is detected or inferred.
- The current multi-tenant case uses deterministic tenant assignment and uniform tenant routing. It does not represent every SaaS tenant distribution.
- Multi-tenant routing labels or namespaces are not equivalent to scalar payload labels. Benchmark clients must keep those fields separate when a run combines tenant routing with scalar-label payload or filter behavior.
- Cold latency depends on cache state, idle window, replica pinning, storage architecture, and service warmup behavior. The idle and warmup rules must stay strict between products.
- Payload search rankings are workload-specific. IDs-only, scalar-label, vector-return, integer-filter, and label-filter runs can produce different winners.
- Cost Pareto results must be read together with recall, latency, payload profile, and readiness constraints rather than as a standalone ranking.
