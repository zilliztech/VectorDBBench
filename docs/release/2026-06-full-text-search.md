# VectorDBBench Full Text Search Release Note

June 2026

Full text search support adds BM25-style text retrieval workloads to VectorDBBench. The goal is to compare database full text search paths with the same benchmark harness used for vector workloads: repeatable datasets, explicit load/search stages, structured result JSONs, and frontend result visualization.

## Context

VectorDBBench has historically focused on dense vector search and vector-oriented cloud cases. That leaves a gap for systems that also expose native text retrieval, sparse search, or BM25 ranking. Users evaluating retrieval systems often need to compare text-only retrieval before deciding whether to use dense vector, sparse vector, hybrid, or reranking layers.

The Full Text Search benchmark covers that text-only layer. It measures end-to-end behavior for indexing raw text, optimizing the backend, searching with BM25-style ranking, and validating retrieval quality against semantic relevance labels from `ir_datasets`.

The benchmark also records payload behavior. Some applications return only document IDs, while others return text fields in the search response. VectorDBBench models both paths through explicit payload profiles so throughput and latency can be interpreted together with the response shape.

## Who we tested this round

This round focuses on full text search support across the following backends:

- Milvus, using its full text search BM25 path.
- Zilliz Cloud, using the cloud full text search path.
- Elasticsearch, using its BM25 text search path.
- Vespa, using BM25 ranking over indexed text fields.
- turbopuffer, using its full text search namespace path.

The benchmark is designed to keep the workload shape consistent while still recording backend-specific behavior. BM25 parameters and analyzer settings are controlled through the backend case config or CLI flags, so users can run either product-default comparisons or explicitly tuned comparisons.

## The new tests we added

### FullTextSearchPerformance

**Purpose.** FullTextSearchPerformance measures BM25-style full text search as a first-class benchmark case. It answers the baseline question: after a backend indexes the same text corpus, what QPS, latency, recall, MRR, and NDCG does it deliver for text queries?

**How it works.** The case loads raw text documents, builds the backend text index, runs the backend optimize path, executes optional serial quality checks, and then runs concurrent search. The result metric records load duration, insert duration, optimize duration, QPS, serial latency, concurrent latency, recall, MRR, NDCG, payload profile, inserted count, and explicit backend parameters.

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

### Full text search datasets and semantic ground truth

**Purpose.** Full text search quality should measure whether a backend returns documents that are semantically relevant to the query, not only whether it reproduces one particular BM25 implementation's exact ranking. This aligns the benchmark with common IR evaluation practice and makes recall, MRR, and NDCG comparable across backend analyzers and ranking implementations.

**How it works.** FTS datasets provide raw text for document insertion and query execution through `ir_datasets`. VectorDBBench loads positive qrels from the same source and uses them as semantic ground truth for serial quality checks. BM25 and analyzer settings are not loaded from dataset manifests; use backend config or CLI flags when a run needs explicit `k1`, `b`, analyzer, or backend-specific search parameters.

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

Committed FTS result JSONs live under `vectordb_bench/results/FullTextSearch/<Backend>/`. Published artifacts are consolidated so each backend directory contains one backend-level JSON with multiple case results in its `results` list, instead of a pile of one-case files. That keeps the release payload readable while preserving the per-dataset and per-payload records needed by the dashboard.

Use the result collector to consolidate split FTS run outputs before committing published examples:

```bash
python -m vectordb_bench.backend.result_collector \
  vectordb_bench/results/FullTextSearch \
  --merge-by-db \
  --task-label fts_standard \
  --replace
```

The default collector behavior still groups normal benchmark files by `run_id`. The `--merge-by-db` mode is intended for curated FTS result publication, where each backend should present the latest benchmark matrix as a single artifact. The Full Text Search frontend reads the latest backend-level result file from each backend directory and displays the committed matrix by backend, dataset, payload profile, load duration, QPS, recall, and latency.

## Caveats

This release note introduces the FTS benchmark path; it is not a complete benchmark report. Detailed raw outputs, machine setup, backend deployment scripts, and rerun notes should live in separate experiment reports when needed.

Important caveats:

- Semantic relevance labels are not the same as exact BM25 implementation correctness. A high quality score means the backend retrieved judged-relevant documents, not that it matched a specific analyzer or scorer bit-for-bit.
- Analyzer behavior can materially affect recall and ranking. Tokenization, lowercase filters, stop words, stemming, token length limits, and field normalization should be recorded with the benchmark run when they are configured explicitly.
- BM25 parameter support differs across products. Some backends expose `k1` and `b`, some expose average field length controls, and some compute or hide those values internally. Use backend config or CLI flags for explicit comparisons; otherwise the run represents product-default behavior.
- IDs-only and text payload runs answer different questions. IDs-only is the cleanest recall and throughput baseline; text payload runs expose response-size overhead.
- Load duration includes backend-specific insert and optimize behavior. Products may differ in whether optimize means force merge, compaction, warmup, or index deployment readiness.
- Result JSONs under `vectordb_bench/results/FullTextSearch` are curated examples for the frontend. They should not be treated as the full historical experiment archive.
