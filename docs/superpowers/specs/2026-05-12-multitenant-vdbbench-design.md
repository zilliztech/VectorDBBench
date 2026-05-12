# Multitenant VDBBench Design

Date: 2026-05-12

## Goal

Add an end-to-end multitenant benchmark case to VDBBench while reusing the existing dataset, insert, readiness, and search infrastructure as much as possible.

The first version targets:

- Turbopuffer: tenant maps to namespace.
- Pinecone: tenant maps to namespace. This follows Pinecone's official serverless multitenancy guidance: one namespace per tenant.
- Zilliz Cloud: tenant maps to a scalar label field using Milvus/Zilliz partition key routing.

The first version measures load time, readiness time, QPS, and latency. It does not measure recall or NDCG.

## Case Shape

Add a new performance-style case named `MultiTenantPerformanceCase`.

Default parameters:

- `dataset_with_size_type`: `Large Cohere (768dim, 10M)`
- `tenant_count`: `1000`
- `tenant_prefix`: `tenant_`
- `tenant_id_width`: `4`
- `tenant_distribution`: `uniform_by_id_mod`
- `measure_recall`: fixed `False` in v1

The case accepts any existing `DatasetWithSizeType`, so smaller datasets can be used for smoke tests. The canonical benchmark remains Cohere 10M with 1000 tenants, matching the external test shape.

Tenant assignment is deterministic:

```python
tenant = f"{tenant_prefix}{row_id % tenant_count:0{tenant_id_width}d}"
```

This avoids an extra mapping file and works with existing VDBBench datasets whose training IDs are integers.

## Load Flow

The load phase should reuse `DatasetManager` and `ConcurrentInsertRunner`.

The insert runner derives tenant labels from each batch's metadata when the case is tenant-aware. Existing scalar label behavior remains unchanged for normal label-filter cases.

Proposed API extension:

```python
insert_embeddings(
    embeddings,
    metadata,
    labels_data=None,
    tenant_labels_data=None,
)
```

Existing clients can ignore `tenant_labels_data`. Target clients use it as follows:

- Turbopuffer groups each batch by tenant and writes each group to `multitenant_namespace_prefix + tenant`.
- Pinecone groups each batch by tenant and upserts each group into `multitenant_namespace_prefix + tenant`.
- Zilliz Cloud stores the tenant as the scalar label value and uses `is_partition_key=True` for the label field.

For namespace-backed systems, grouped writes may be smaller than the external repo's pre-split per-tenant parquet loader. This is acceptable for v1 because it keeps the implementation inside VDBBench's existing streaming batch path. A later optimization can add per-tenant pre-splitting if write efficiency becomes the bottleneck.

## Optimize And Readiness

Readiness should be part of v1 so runs are reproducible and do not depend on manually preloaded state.

- Zilliz Cloud uses the existing collection optimize/readiness behavior.
- Turbopuffer polls readiness across all touched tenant namespaces and reports fully indexed only when all relevant namespaces report zero unindexed bytes.
- Pinecone polls index stats/readiness across namespaces. Namespace readiness is complete when every configured tenant namespace has at least the expected inserted count for that tenant and the existing LSN freshness probe has caught up after the final write.

The set of touched tenants is deterministic from `tenant_count`. For partial-duration load cases, clients may track touched tenants during insert; the default full-load case can check all configured tenants.

## Search Flow

The search phase should reuse `SerialSearchRunner` and `MultiProcessingSearchRunner` with tenant-aware extensions.

Each query selects a tenant uniformly at random from the configured tenant list. The query vector sequence can continue to use existing test vectors; the tenant selection is independent of query vector.

Proposed API extension:

```python
search_embedding(query, k=100, payload_profile=PayloadProfile.IDS_ONLY, tenant=None)
```

Target backend behavior:

- Turbopuffer resolves `tenant` to `multitenant_namespace_prefix + tenant` and queries that namespace.
- Pinecone resolves `tenant` to `multitenant_namespace_prefix + tenant` and passes that as the Pinecone namespace.
- Zilliz Cloud keeps a single collection and converts `tenant` to a scalar filter against the label field.

Serial search records latency only. Recall and NDCG are set to `0` in v1 because tenant-local ground truth is not generated.

## Backend Requirements

### Turbopuffer

Add namespace-aware insert, search, drop, and readiness support.

The client should cache namespace handles by tenant within a process to avoid reconstructing them on every query.

Add a `multitenant_namespace_prefix` DB config. For Turbopuffer, this is separate from the existing single-case `namespace` config so normal single-namespace benchmarks keep their current behavior.

### Pinecone

Add namespace-aware upsert and query behavior.

Drop-old behavior should delete only benchmark tenant namespaces when possible, instead of deleting unrelated namespaces in the same index. This avoids damaging user data when an index is shared.

Add a `multitenant_namespace_prefix` DB config. Pinecone has no current namespace config in VDBBench, so this field is used only when the multitenant case is active.

### Zilliz Cloud

Reuse the Milvus scalar-label path. For the multitenant case, the label field must be created with `is_partition_key=True`.

The case should fail fast for Zilliz Cloud if partition-key routing is disabled, because otherwise the benchmark would not represent the intended multitenant layout.

## Metrics

Reuse existing performance metrics:

- `insert_duration`
- `optimize_duration`
- `load_duration`
- `qps`
- `conc_num_list`
- `conc_qps_list`
- `conc_latency_p99_list`
- `conc_latency_p95_list`
- `conc_latency_avg_list`
- `serial_latency_p99`
- `serial_latency_p95`

Set `recall=0` and `ndcg=0` for v1. Result metadata must include the multitenant parameters so different tenant counts and datasets are not compared blindly.

## CLI And Config

Add the case to the same CLI/custom-case pattern used by `CloudPayloadSearchCase`.

Expected custom case parameters:

- `dataset_with_size_type`
- `tenant_count`
- `tenant_prefix`
- `tenant_id_width`

The frontend can be added later. CLI support is enough for the first implementation.

## Testing

Add focused unit tests:

- Tenant label generation from integer IDs.
- `MultiTenantPerformanceCase` default construction and custom dataset/tenant count construction.
- Insert runner derives `tenant_labels_data` from batch metadata for tenant-aware cases.
- Namespace-backed fake client receives grouped insert calls per tenant.
- Search runner passes a valid tenant to `search_embedding`.
- Zilliz multitenant case rejects `use_partition_key=False`.
- Existing non-multitenant cases still call clients without tenant context.

## Out Of Scope

- Tenant-local recall and NDCG.
- Zipfian or hot-tenant traffic distributions.
- Noisy-neighbor isolation tests.
- Per-tenant parquet pre-splitting for optimized bulk load.
- Frontend UI integration.
- Support for DBs beyond Turbopuffer, Pinecone, and Zilliz Cloud.
