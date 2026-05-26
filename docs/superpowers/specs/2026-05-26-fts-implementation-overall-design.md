# FTS Implementation Overall Design

**Date:** 2026-05-26
**Status:** Draft for review
**Scope:** Architecture and lifecycle boundaries for BM25 full-text retrieval support in VectorDBBench.

## Decision

Implement FTS as a first-class BM25 retrieval workload, not as a fully separate benchmark product and not as a forced extension of vector dataset internals.

The implementation should separate FTS-specific semantics while reusing the existing benchmark lifecycle wherever the mechanics are the same:

- Keep FTS corpus, query, qrel, analyzer, schema, and backend text-search behavior separate.
- Reuse the shared performance lifecycle: load, optimize, serial search, concurrent search, metric/result output.
- Reuse the existing result JSON fields where possible.
- Avoid broad retrieval-framework refactors until Milvus, Elasticsearch, Vespa, and Turbopuffer expose the real common shape.

## Workload Boundary

The first implementation draft covers native BM25-ranked top-k retrieval only.

In scope:

- Milvus BM25 sparse retrieval.
- Elasticsearch BM25 text retrieval.
- Vespa BM25 text retrieval.
- Turbopuffer BM25 text retrieval.
- English MS MARCO Passage.
- BEIR HotpotQA.
- Existing performance metrics: QPS, serial latency, recall, NDCG, MRR, insert duration, optimize duration, load duration.
- Optional response payload profile for FTS searches: `ids_only` by default, with an opt-in text payload mode that retrieves the matched document text.

Out of scope:

- ClickHouse token/boolean full-text search.
- Phrase search.
- Boolean query search.
- Fuzzy or prefix search.
- Dense+sparse hybrid search.
- Text-filter benchmarks.
- Externally supplied sparse vectors or SPLADE-style retrieval.
- New `*_at_k` metric fields.
- Persisting returned document text into benchmark result JSON.

ClickHouse should be tracked as a later token/boolean FTS benchmark, not included in the first BM25 matrix.

## Layers That Stay FTS-Specific

### Dataset Model

Keep FTS dataset types separate from the existing vector dataset types.

FTS needs:

- Text corpus documents.
- Text queries.
- Qrels-based relevance judgments.
- qrel-preserving corpus caps.
- `ir_datasets` streaming.
- Stable external query and document IDs, normalized as strings.

Vector datasets assume:

- Embedding dimensions.
- Vector metric types.
- Parquet train/test vectors.
- nearest-neighbor ground truth.
- Optional scalar-label/filter files.

Because these assumptions are materially different, FTS should keep dedicated types such as `FtsQuery`, `FtsDocument`, `FtsGroundTruth`, translators, and dataset managers. The immediate cleanup should correct the current FTS dataset path rather than merge it into `DatasetManager`.

Required dataset fixes:

- Use `qrels_iter()` as relevance ground truth.
- Treat query IDs and document IDs as strings end-to-end.
- Enforce `Small`, `Medium`, and `Large` corpus size semantics.
- Build capped corpora by including positive-qrel documents first, then deterministic filler.

### Backend Text Search Semantics

Keep backend-native FTS schema and query logic inside each backend adapter.

Examples:

- Milvus: BM25 function from text to sparse vector, sparse index config, analyzer config.
- Elasticsearch: `text` field mapping, BM25 similarity, `match` query, refresh behavior.
- Vespa: indexed text field, `index: enable-bm25`, rank profile using `bm25(text)`.
- Turbopuffer: text field with `full_text_search`, BM25 `rank_by`, metadata index readiness.

These should not be squeezed into vector index configuration models. The common contract should be small: insert text documents, search text queries, and return top-k document IDs.

### Metric Calculation Semantics

Reuse the output fields, but keep the internal FTS relevance calculations separate.

Vector quality metrics compare returned IDs against ordered nearest-neighbor ground truth. FTS quality metrics compare returned IDs against qrels. The JSON field names can stay as `recall`, `ndcg`, and `mrr`, with the configured `k` recorded in task configuration.

### Response Payload Semantics

FTS should support an explicit response payload profile. Retrieving only IDs is not representative of every production FTS workload; many applications search text and then return the actual matched text, snippets, or document body.

The first FTS implementation should support:

- `ids_only`: default. Search requests return only document IDs and scores where the backend requires scores internally.
- `text`: opt-in. Search requests also ask the backend to return the stored document text field for each top-k hit.

The benchmark should still compute recall, NDCG, and MRR from returned document IDs. Text payloads are requested and transferred to measure latency/QPS impact, but they should not be stored in result JSON. Result metadata should record the selected payload profile and an estimated response payload size per query when available.

This follows the design shape from PR 775's cloud payload work (https://github.com/zilliztech/VectorDBBench/pull/775): payload is a search response profile threaded through cases, runners, backend search calls, and result metadata. If that PR is merged first, reuse or extend its payload-profile mechanism rather than creating a second unrelated one. For FTS, the payload profile must be text-aware rather than vector-aware.

### Case Identity

Keep `CaseLabel.FullTextSearchPerformance` as a distinct case label.

This prevents BM25 retrieval from being presented as equivalent to dense vector search and gives the UI/CLI a clear capability gate. The case definitions should be thin, dataset-driven definitions rather than a separate execution stack.

### Optional Backend Capability

FTS should be an explicit backend capability, not a required method on every vector database client.

Supported first-draft capability set:

- Milvus.
- Elasticsearch.
- Vespa.
- Turbopuffer.

Unsupported for the BM25 case:

- ClickHouse.
- Any backend that does not declare the BM25 text-search capability.

The API can use either a capability flag/support map or a protocol-style boundary. Unsupported clients should fail at task construction or UI selection time, not halfway through a benchmark run.

## Layers That Reuse Existing Lifecycle

### Performance Orchestration

FTS should run through the same high-level lifecycle as vector performance benchmarks:

1. Prepare the dataset.
2. Load records.
3. Optimize/build index if the backend requires it.
4. Run serial search.
5. Run concurrent search.
6. Produce the shared `Metric` result.

The current `_run_fts_perf_case()` path duplicates too much of the vector performance lifecycle. Replace it with one performance runner that dispatches only mode-specific operations:

- Load adapter: vector embeddings vs text documents.
- Search payload adapter: vectors vs text queries.
- Ground-truth adapter: neighbor IDs vs qrels.
- Metric adapter: vector metrics vs qrels metrics.

### Task Stages And Result Schema

Reuse existing task stages and result output:

- `TaskStage.LOAD`
- `TaskStage.SEARCH_SERIAL`
- `TaskStage.SEARCH_CONCURRENT`
- `Metric`
- existing result JSON fields

Do not add duplicate `recall_at_k`, `ndcg_at_k`, or `mrr_at_k` fields. The configured `k` is already part of the task configuration.

### Search Runners

Keep serial and multiprocessing search runners shared, but make the workload mode explicit.

The current `search_fulltext` flag and string autodetection are a useful prototype bridge, but the implementation should move toward an explicit mode such as:

```python
WorkloadKind = VECTOR | FULL_TEXT_BM25
```

or a small search adapter object. The runner should not infer search semantics from whether the first query payload is a string.

Search runners should also pass the selected response payload profile to backend search calls. The runner may continue to consume only document IDs for metric calculation, but the backend call must actually request the configured payload so latency and QPS reflect the chosen response envelope.

### UI, CLI, And Task Assembly

FTS should be selectable through the normal task/config path for backends that declare BM25 FTS support.

The first-draft UI and CLI should not become long-term Milvus-only side paths. They should expose the FTS case only for the supported BM25 backends and should preserve backend, dataset, size tier, and workload labels in task/result metadata.

The UI and CLI should expose response payload as a normal FTS case parameter:

- default: `ids_only`
- optional: `text`

This should not create separate public case classes such as `FTSWithTextPayload`; it is a workload parameter like `k` or concurrency.

## Insert Runner Strategy

Keep `SerialFtsInsertRunner` separate for the first cleanup, but avoid duplicating accounting and lifecycle logic unnecessarily.

Reasons to keep it separate initially:

- Text ingestion loads `(doc_id, text)` rather than vectors.
- `ir_datasets` streaming has different memory and pickling constraints.
- Backend insert methods may need text-specific batching or readiness behavior.

Reasons to revisit later:

- Retry handling.
- Batch timing.
- inserted-record accounting.
- timeout behavior.
- error reporting.

After the first BM25 backend set is implemented, reassess whether a shared insert runner with payload adapters would reduce real duplication.

## Minimal Interface Changes

Add a workload discriminator:

```python
WorkloadKind.VECTOR
WorkloadKind.FULL_TEXT_BM25
```

Expose the workload from the case or dataset manager.

Add a narrow FTS backend contract:

```python
insert_documents(texts: list[str], doc_ids: list[str]) -> tuple[int, str | None]
search_documents(
    query_text: str,
    k: int,
    payload_profile: FtsPayloadProfile = FtsPayloadProfile.IDS_ONLY,
) -> list[str]
```

The exact return signature should follow existing VectorDBBench conventions, but document IDs must be normalized for metric comparison.

The text payload mode should keep the return value as document IDs for metric compatibility. Backend adapters should request the text field and discard or ignore the returned text after the request completes unless a later design adds explicit payload validation.

Dataset managers should expose enough common lifecycle data for the shared runner:

- workload kind
- record iterator
- query payloads
- ground truth
- dataset size label
- dataset source label
- estimated text payload bytes per query, if available

The case config or search adapter should expose the selected response payload profile. Payload choice is a workload parameter, not a dataset identity.

If PR 775's `Metric.payload_profile` and `Metric.payload_estimated_bytes_per_query` fields are present, FTS should reuse them. If not, add equivalent result metadata fields, not duplicate metric families.

The existing vector path should remain the default behavior. FTS changes should be additive and capability-gated.

## Migration Plan

1. Fix FTS dataset correctness:
   - string IDs
   - qrels ground truth
   - size caps
   - MS MARCO and HotpotQA options
2. Introduce workload/capability boundaries:
   - `WorkloadKind`
   - BM25 FTS backend capability gate
   - explicit search mode instead of `search_fulltext` inference
   - FTS response payload profile with `ids_only` and `text`
3. Fold FTS into the shared performance lifecycle:
   - one orchestration path
   - mode-specific load/search/metric adapters
   - shared result schema
   - payload profile threaded through serial and concurrent search
4. Implement backend adapters:
   - Milvus cleanup
   - Elasticsearch
   - Vespa
   - Turbopuffer
5. Wire UI/CLI/task assembly:
   - supported backend gating
   - dataset size selection
   - result labels
6. Add focused tests:
   - dataset caps
   - qrels loading
   - string ID normalization
   - FTS case generation for supported backends
   - unsupported backend gating
   - metric calculation from qrels
   - payload profile propagation and unsupported payload gating

## Risks

### Premature Generalization

A broad retrieval abstraction could hide backend-specific analyzer and ranking behavior before the implementation has enough backend coverage. Keep the shared lifecycle small and observable.

### Vector Regression

Changing the runner lifecycle can affect existing vector benchmarks. Preserve vector defaults and add tests around vector task creation and the existing performance path.

### ID Normalization

Moving from integer IDs to string IDs affects Milvus primary keys, result comparison, and existing FTS code paths. Normalize at dataset and metric boundaries and document backend-specific storage choices.

### Unsupported Backend Leakage

If UI/CLI/task assembly does not gate FTS support, unsupported backends may fail only after loading starts. Capability checks should happen before task creation whenever possible.

### Analyzer Comparability

Backends expose different analyzer defaults and tokenizer controls. The first benchmark should document defaults and avoid claiming analyzer-normalized semantic equivalence across systems.

### Payload Comparability

Text payload sizes vary by dataset and backend response encoding. Use the payload profile to measure realistic response-envelope cost, but treat estimated bytes as approximate metadata unless backend-specific exact byte accounting is added later.

## Case Coverage Discussion Point

The next design decision is which FTS cases to expose publicly. The case matrix should balance useful coverage against benchmark explosion across:

- backend: Milvus, Elasticsearch, Vespa, Turbopuffer
- dataset: MS MARCO, HotpotQA
- corpus size: Small, Medium, Large
- query concurrency and configured `k`
- response payload profile: `ids_only` vs `text`
- backend-specific analyzer/config variants

The default should favor a compact, comparable BM25 retrieval matrix first, with larger and backend-tuning-heavy cases available as opt-in extensions.
