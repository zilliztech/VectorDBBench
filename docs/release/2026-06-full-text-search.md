# VectorDBBench Full Text Search Release Note

June 2026

Full text search support adds BM25-style text retrieval workloads to VectorDBBench. The goal is to compare database full text search paths with the same benchmark harness used for vector workloads: repeatable datasets, explicit load/search stages, structured result JSONs, and frontend result visualization.

## Context

VectorDBBench has historically focused on dense vector search and vector-oriented cloud cases. That leaves a gap for systems that also expose native text retrieval, sparse search, or BM25 ranking. Users evaluating retrieval systems often need to compare text-only retrieval before deciding whether to use dense vector, sparse vector, hybrid, or reranking layers.

The Full Text Search benchmark covers that text-only layer. It measures end-to-end behavior for indexing raw text, optimizing the backend, searching with BM25-style ranking, and validating recall against mathematical ground truth. The benchmark intentionally separates this from semantic relevance labels: recall is computed against generated BM25 top-k ground truth, not human relevance judgments.

The benchmark also records payload behavior. Some applications return only document IDs, while others return text fields in the search response. VectorDBBench models both paths through explicit payload profiles so throughput and latency can be interpreted together with the response shape.

## Who we tested this round

This round focuses on full text search support across the following backends:

- Milvus, using its full text search BM25 path.
- Zilliz Cloud, using the cloud full text search path.
- Elasticsearch, using its BM25 text search path.
- Vespa, using BM25 ranking over indexed text fields.
- turbopuffer, using its full text search namespace path.

The benchmark is designed to keep the workload shape consistent while still recording backend-specific behavior. BM25 parameters and analyzer settings are read from dataset manifests when available, applied when the backend exposes matching controls, and recorded as unapplied parameters when a backend does not expose the same control.

## The new tests we added

### FullTextSearchPerformance

**Purpose.** FullTextSearchPerformance measures BM25-style full text search as a first-class benchmark case. It answers the baseline question: after a backend indexes the same text corpus, what QPS, latency, and mathematical recall does it deliver for text queries?

**How it works.** The case loads raw text documents, builds the backend text index, runs the backend optimize path, executes optional serial recall checks, and then runs concurrent search. The result metric records load duration, insert duration, optimize duration, QPS, serial latency, concurrent latency, recall, payload profile, inserted count, and additional parameters such as manifest BM25 settings.

Example: run MS MARCO small on Milvus with IDs-only responses.

```bash
vectordbbench milvusfts \
  --case-type FTSBm25Performance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --uri "$MILVUS_URI" \
  --payload-profile ids_only \
  --load-concurrency 0 \
  --num-concurrency 40,80 \
  --task-label fts-milvus-msmarco-small-ids
```

### Full text search datasets and math ground truth

**Purpose.** Full text search recall should measure whether an implementation returns the mathematically expected BM25 neighbors for the indexed corpus. Human relevance labels are useful for IR evaluation, but they are not a direct correctness target for a database BM25 implementation.

**How it works.** FTS datasets provide raw text for document insertion and query execution. The ground-truth artifacts provide top-k neighbor IDs generated under a declared BM25/analyzer contract. Each dataset artifact can include a build manifest with BM25 parameters such as `k1`, `b`, and `avgdl`, plus analyzer settings. VectorDBBench loads those values before backend initialization so index construction can use the dataset contract where the backend supports it.

Example: use a larger dataset while keeping the same FTS case type.

```bash
vectordbbench elasticcloudhnsw \
  --case-type FTSBm25Performance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --password "$ELASTIC_PASSWORD" \
  --payload-profile ids_only \
  --load-concurrency 0 \
  --num-concurrency 40,80 \
  --task-label fts-elastic-hotpotqa-medium-ids
```

### Payload-aware full text search

**Purpose.** Payload-aware FTS measures the cost of returning text fields, not only document IDs. This matters for applications where search results immediately include snippets, source text, or other fields needed by downstream ranking and display layers.

**How it works.** The FTS case supports `payload_profile` values including `ids_only` and `text`. IDs-only runs are the recall baseline because they can run serial search against the ground truth without paying text-return overhead. Text payload runs measure concurrent search throughput and latency for larger response bodies; they can skip serial recall when the same indexed namespace or collection has already been validated by the IDs-only run.

Example: run Vespa with text payload responses and concurrent search only.

```bash
vectordbbench vespa \
  --case-type FTSBm25Performance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --payload-profile text \
  --load-concurrency 0 \
  --skip-search-serial \
  --num-concurrency 40,80 \
  --task-label fts-vespa-msmarco-medium-text
```

## Result artifacts

Committed FTS result JSONs live under `vectordb_bench/results/FullTextSearch/<Backend>/`. The Full Text Search frontend page reads that directory directly and displays the committed benchmark matrix by backend, dataset, payload profile, load duration, QPS, recall, and latency.

## Caveats

This release note introduces the FTS benchmark path; it is not a complete benchmark report. Detailed raw outputs, machine setup, backend deployment scripts, and rerun notes should live in separate experiment reports when needed.

Important caveats:

- Mathematical BM25 ground truth is not the same as semantic relevance evaluation. A high recall score means the backend matched the declared BM25 ranking contract, not that it matched human relevance labels.
- Analyzer behavior can materially affect recall and ranking. Tokenization, lowercase filters, stop words, stemming, token length limits, and field normalization should be recorded with each dataset manifest and backend result.
- BM25 parameter support differs across products. Some backends expose `k1` and `b`, some expose average field length controls, and some compute or hide those values internally. VectorDBBench records applied and unapplied parameters so results can be interpreted correctly.
- IDs-only and text payload runs answer different questions. IDs-only is the cleanest recall and throughput baseline; text payload runs expose response-size overhead.
- Load duration includes backend-specific insert and optimize behavior. Products may differ in whether optimize means force merge, compaction, warmup, or index deployment readiness.
- Result JSONs under `vectordb_bench/results/FullTextSearch` are curated examples for the frontend. They should not be treated as the full historical experiment archive.
