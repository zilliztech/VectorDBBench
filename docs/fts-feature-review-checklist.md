# FTS Feature Review Checklist

**Date:** 2026-05-25
**Last Updated:** 2026-05-26
**Scope Reviewed:** Initial full-text search support in VectorDBBench
**Current Conclusion:** The implementation is best described as an initial Milvus BM25 sparse retrieval benchmark. The first implementation draft should focus on native BM25-ranked top-k retrieval for Milvus, ElasticCloud, Vespa, and Turbopuffer. ClickHouse should be excluded from the first BM25 draft because its official full-text search API is token/boolean oriented rather than native BM25-ranked retrieval. The initial dataset matrix should use English MS MARCO Passage and BEIR HotpotQA; Chinese mMARCO should be tracked as a later multilingual/analyzer stress dataset.

## Review Summary

The current FTS feature adds a separate benchmark path for Milvus BM25 over MS MARCO text data. It does not implement broader full-text search semantics such as phrase search, boolean query parsing, fuzzy or prefix search, text filters, match-expression benchmarking, externally supplied sparse vectors, SPLADE-style retrieval, or dense+sparse hybrid search.

Several correctness issues should be resolved before the feature is presented as a stable benchmark. The highest-risk problems are dataset size mismatch, wrong relevance source for metrics, missing runtime dependency in Docker requirements, and UI/backend configuration mismatch.

## Confirmed Target Requirements

- Required initial BM25 retrieval backends:
  - Milvus
  - ElasticCloud
  - Vespa
  - Turbopuffer
- Future token/boolean FTS backend:
  - ClickHouse
- Required initial datasets:
  - MS MARCO Passage, English
  - HotpotQA via BEIR
- Future dataset candidate:
  - Chinese mMARCO for multilingual tokenizer/analyzer coverage

The current PR does not meet these requirements yet. It currently wires one MS MARCO-style translator and one Milvus BM25 implementation path.

## Scope Boundary

The first implementation draft is a BM25-ranked retrieval benchmark:

- Input: natural-language text queries from MS MARCO or HotpotQA.
- Backend behavior: native BM25 or BM25-equivalent scoring over a text field.
- Output: top-k document IDs ranked by the backend's native BM25 score.
- Metrics: existing VectorDBBench `qps`, `serial_latency_p95`, `serial_latency_p99`, `recall`, `ndcg`, and `mrr`; `k` is already recorded in `task_config.case_config.k`, so no duplicate `*_at_k` fields are needed.
- Payload: first implementation runs IDs-only. Text payload retrieval should be added after PR 775's shared payload mechanism lands, reusing that infrastructure rather than creating an FTS-only payload abstraction.

The first implementation draft is not a generic FTS benchmark:

- No phrase search benchmark.
- No boolean query benchmark.
- No fuzzy or prefix search benchmark.
- No dense+sparse hybrid benchmark.
- No text-filter benchmark.
- No externally supplied sparse-vector benchmark.

ClickHouse is explicitly out of scope for the first BM25 retrieval draft. Official ClickHouse full-text search uses `TYPE text` indexes with token/boolean functions such as `hasAnyTokens`, `hasAllTokens`, and `hasPhrase`; it does not expose a documented native BM25 top-k retrieval API comparable to ElasticCloud, Vespa, Turbopuffer, or Milvus BM25. Track ClickHouse as a later token/boolean FTS benchmark with performance-oriented metrics.

Future FTS response payload should be measured as part of the search envelope, not as a separate metric family. Metrics should still be computed from returned document IDs. The selected payload profile and estimated payload bytes per query should be recorded as metadata when available, but returned document text should not be persisted in result JSON.

## Dataset Design Decision

- Use corpus-size tiers, not native ir-datasets split names, for benchmark labels.
  - `Small`: 100K documents
  - `Medium`: 1M documents
  - `Large`: full native corpus
- Use a qrel-preserving cap for `Small` and `Medium`.
  - First include all documents with positive relevance judgments for the selected query split.
  - Then fill the remaining corpus deterministically from the native corpus order until the target size is reached.
  - Reject any cap that is smaller than the number of required qrel documents.
- Use qrels as relevance ground truth, never `scoreddocs`.
- Normalize FTS query and document IDs as strings end-to-end.
- Initial source IDs:
  - MS MARCO: `msmarco-passage/dev/small` for the first query/qrels split, with corpus inherited from `msmarco-passage`.
  - HotpotQA: `beir/hotpotqa/test` for the first query/qrels split, with corpus inherited from `beir/hotpotqa`.
- Initial dataset options:
  - `MS MARCO Small (100K documents)`
  - `MS MARCO Medium (1M documents)`
  - `MS MARCO Large (8.8M documents)`
  - `HotpotQA Small (100K documents)`
  - `HotpotQA Medium (1M documents)`
  - `HotpotQA Large (5.2M documents)`
- Small and Medium are default visible tiers. Large is available as an advanced or opt-in tier.
- Defer `nano-beir/hotpotqa`.
  - It is useful for smoke tests, but should not be exposed as a benchmark dataset tier in the first public matrix.
- Defer Chinese mMARCO.
  - It is useful for later multilingual analyzer/tokenizer coverage, but the initial cross-backend benchmark should use English MS MARCO to reduce analyzer normalization complexity.

## Target Coverage Checklist

- [x] Confirm the exact Elasticsearch target in VectorDBBench terms.
  - Evidence: the codebase has `DB.ElasticCloud`, `DB.AliyunElasticsearch`, and `DB.TencentElasticsearch`, but the confirmed requirement says "Elasticsearch".
  - Decision: Use `DB.ElasticCloud` for the first BM25 implementation. Aliyun Elasticsearch and Tencent Elasticsearch are out of first-draft scope unless explicitly added later.
  - Acceptance: The design doc and UI list `DB.ElasticCloud` as the Elasticsearch-compatible backend covered by FTS.

- [ ] Add FTS backend support for ElasticCloud.
  - Evidence: current Elasticsearch client paths are vector-search oriented and do not implement `insert_documents()` or `search_documents()`.
  - Impact: ElasticCloud is one of the confirmed target backends and cannot run the current FTS case.
  - Acceptance: ElasticCloud can load the shared text corpus, run text queries, and report recall, NDCG, MRR, serial latency, and QPS through the same FTS case runner.

- [ ] Add FTS backend support for Vespa.
  - Evidence: current Vespa config and wrapper are HNSW/vector oriented.
  - Impact: Vespa is one of the confirmed target backends and needs a text retrieval schema/ranking path.
  - Acceptance: Vespa can load the shared text corpus, run text queries, and report the shared FTS metrics.

- [ ] Add FTS backend support for Turbopuffer.
  - Evidence: current Turbopuffer wrapper is present as `DB.TurboPuffer`, but the FTS path is not wired for it.
  - Impact: Turbopuffer is one of the confirmed target backends and needs a text retrieval implementation path.
  - Acceptance: Turbopuffer can load the shared text corpus, run text queries, and report the shared FTS metrics using its supported text search API.

- [ ] Defer ClickHouse to a token/boolean FTS benchmark.
  - Evidence: current ClickHouse wrapper is vector-search oriented and does not implement the FTS document/search methods.
  - Evidence: official ClickHouse text search uses `TYPE text` indexes and token/boolean functions, not a documented native BM25 top-k scorer.
  - Impact: Forcing ClickHouse into the BM25 retrieval matrix would compare a synthetic scoring path against native BM25 systems.
  - Acceptance: The first BM25 retrieval matrix excludes ClickHouse, and a separate follow-up tracks ClickHouse token/boolean FTS semantics and metrics.

- [ ] Add MS MARCO English dataset support.
  - Evidence: current code only defines `MSMarcoTranslator` for `msmarco-passage/dev/small`.
  - Impact: MS MARCO is the initial English FTS baseline, but the current implementation advertises 100K docs without enforcing the cap and uses `scoreddocs` instead of qrels.
  - Acceptance: The dataset registry includes MS MARCO `Small`, `Medium`, and `Large` options with exact source ID(s), qrel-preserving corpus caps, qrels-based ground truth, stable string IDs, and tests.

- [ ] Add HotpotQA dataset support.
  - Evidence: no HotpotQA translator, case, or registry entry exists in the current FTS path.
  - Impact: HotpotQA is a confirmed target dataset and cannot run through the current benchmark.
  - Acceptance: The dataset registry includes HotpotQA `Small`, `Medium`, and `Large` options with exact source ID(s), qrel-preserving corpus caps, qrels-based ground truth, stable string IDs, and tests.

- [ ] Track Chinese mMARCO as a future dataset.
  - Evidence: Chinese mMARCO is useful for exposing tokenizer/analyzer differences in multilingual FTS, but it adds backend normalization complexity.
  - Impact: Adding it later keeps the first benchmark matrix focused while preserving the multilingual requirement as a follow-up.
  - Acceptance: The design docs list Chinese mMARCO as future work, not as part of the initial required backend x dataset matrix.

- [ ] Define the BM25 backend x dataset benchmark matrix.
  - Evidence: target coverage spans Milvus, ElasticCloud, Vespa, Turbopuffer, and two initial datasets; current UI exposes one FTS case.
  - Impact: Without an explicit matrix, results may be incomplete or hard to compare.
  - Acceptance: The UI/CLI can express each intended backend/dataset pair, Small and Medium are default visible tiers, Large is advanced or opt-in, and the result output preserves backend and dataset labels.

- [ ] Add an FTS response payload option after PR 775 lands.
  - Evidence: PR 775 (https://github.com/zilliztech/VectorDBBench/pull/775) adds response payload profiles for cloud benchmarks and threads payload choice through cases, runners, backend search calls, and result metadata.
  - Impact: FTS often retrieves matched text after search; an IDs-only benchmark can understate latency and QPS cost for production text-search workloads.
  - Acceptance: First implementation does not add an FTS-only payload abstraction. After PR 775 lands, FTS supports `ids_only` and `text` payload profiles by reusing the shared payload mechanism; serial and concurrent search pass the profile to backend `search_documents()` calls; supported backends request the text field in `text` mode; metric calculation still uses IDs; result metadata records the selected payload profile and estimated payload bytes per query when available.

## Scope Checklist

- [ ] Rename or document the current implementation as "Milvus BM25 sparse retrieval" until the full target matrix is implemented.
  - Evidence: `MilvusFtsConfig` uses `SPARSE_INVERTED_INDEX` and `BM25`; Milvus schema creates a `FunctionType.BM25` function from `text` to `sparse_vector`.
  - Impact: Avoids implying support for the newly confirmed backend and dataset matrix before it exists.
  - Acceptance: UI, CLI help, docs, and PR description state the current limitation and the target coverage separately.

- [ ] Treat Milvus-only coverage as a current gap, not the final target.
  - Evidence: only Milvus implements `insert_documents()` and `search_documents()`; UI config maps `CaseLabel.FullTextSearchPerformance` only under `DB.Milvus`; CLI only registers `MilvusFTS`.
  - Impact: The confirmed first BM25 target requires ElasticCloud, Vespa, and Turbopuffer in addition to the existing Milvus path.
  - Acceptance: The issue tracker or PR checklist has explicit tasks for each required backend.

- [ ] Document that this is sparse-only search, not dense+sparse hybrid search.
  - Evidence: FTS mode sets `dim = 0`; Milvus creates `doc_id`, `text`, and `sparse_vector`, with no dense vector field.
  - Impact: Avoids conflating this benchmark with Milvus hybrid search APIs and rankers such as RRF or weighted ranking.
  - Acceptance: Feature docs and PR text state that dense vector benchmarking and BM25 sparse retrieval are separate modes.

## Correctness Checklist

- [ ] Enforce or correct the advertised "100K documents" dataset size.
  - Evidence: `MSMarcoFts` labels size `100_000`, but `FtsDocumentIterator` streams `docs_iter()` until exhaustion. The ir-datasets page for `msmarco-passage/dev/small` says it inherits 8,841,823 docs from `msmarco-passage`.
  - Impact: A case advertised as 100K can load the full MS MARCO passage corpus, changing load time, index size, QPS, and reproducibility.
  - Acceptance: Either cap inserted docs at `self.data.size` or rename the case to reflect the actual corpus size.

- [ ] Use qrels, not scoreddocs, as relevance ground truth.
  - Evidence: `MSMarcoTranslator.load_ground_truth()` reads `dataset.scoreddocs_iter()`. MS MARCO `scoreddocs` are BM25 top-1000 candidate documents for re-ranking; qrels are the relevance judgments.
  - Impact: Recall, NDCG, and MRR are currently measured against BM25 candidate lists rather than relevance labels, making the reported quality metrics misleading.
  - Acceptance: Ground truth loads from `qrels_iter()` and tests verify metric inputs for at least one query.

- [ ] Generalize FTS query and document IDs beyond integers.
  - Evidence: `FtsQuery.query_id`, `FtsDocument.doc_id`, and `FtsGroundTruth` are typed as `int`, and `MSMarcoTranslator` casts IDs with `int(...)`.
  - Impact: MS MARCO, HotpotQA, and future mMARCO source IDs should be treated as external dataset identifiers, and backend APIs may preserve string IDs.
  - Acceptance: FTS IDs use a stable `str` or `str | int` type end-to-end, and all result/ground-truth comparisons normalize IDs consistently.

- [ ] Add `ir_datasets` to Docker/runtime requirements.
  - Evidence: `pyproject.toml` includes `ir_datasets`, but `install/requirements_py3.11.txt` does not. The Dockerfile installs from `install/requirements_py3.11.txt`, and `vectordb_bench/backend/data_source.py` imports `ir_datasets` at module import time.
  - Impact: The app container can fail with `ModuleNotFoundError` before any benchmark starts.
  - Acceptance: `install/requirements_py3.11.txt` includes `ir_datasets`, and a clean Docker or requirements install can import `vectordb_bench.backend.data_source`.

- [ ] Fix FTS UI/backend config wiring for analyzer parameters.
  - Evidence: UI sends `analyzer_max_len`, but backend expects `analyzer_max_token_length`. Milvus schema hardcodes `analyzer_params={"type": "english"}` on the `text` field, while `MilvusFtsConfig.index_param()` builds analyzer params separately.
  - Impact: User-configured analyzer settings can be ignored or applied to the wrong object.
  - Acceptance: Field names match end-to-end, and the generated Milvus schema/index params reflect selected tokenizer, lowercase, max token length, and stop words.

- [ ] Guard FTS case selection for non-Milvus databases.
  - Evidence: `get_case_config_inputs()` indexes `CASE_CONFIG_MAP[db][CaseLabel.FullTextSearchPerformance]`, but only `DB.Milvus` has that key.
  - Impact: Selecting the FTS case with another active DB can raise `KeyError` or create an unsupported task.
  - Acceptance: The UI hides FTS unless Milvus is selected, or shows a clear unsupported state without raising.

## Design Checklist

- [ ] Decide whether to keep the forked FTS lifecycle or introduce a retrieval-case abstraction.
  - Evidence: The implementation adds `FtsDatasetManager`, `FtsDataset`, `FtsDocumentIterator`, `SerialFtsInsertRunner`, `_run_fts_perf_case`, `_init_fts_search_runner`, and `search_fulltext` flags.
  - Impact: The current split works for a narrow first backend but may become hard to extend for additional retrieval modalities and backends.
  - Acceptance: The PR includes an explicit design decision: either keep FTS separate as a narrow Milvus path, or define a shared retrieval lifecycle for corpus, queries, ground truth, insert, search, and metrics.

- [ ] Keep dataset adapters separate from vector parquet datasets, but avoid duplicating benchmark lifecycle unnecessarily.
  - Evidence: Existing `DatasetManager` is vector/parquet-oriented and assumes dimensions, metric types, train/test vectors, neighbor files, and scalar-label files. FTS uses corpus documents, queries, and qrels.
  - Impact: Some separation is justified; excessive separation increases maintenance cost.
  - Acceptance: Dataset-specific loading remains adapter-based, while common runner behavior is shared where it is genuinely common.

- [ ] Define the intended extension model before adding more FTS backends or query types.
  - Evidence: Current API adds Milvus-specific `insert_documents()` and `search_documents()` without a broader capability model.
  - Impact: Adding ElasticCloud, Vespa, Turbopuffer, MS MARCO, HotpotQA, and later ClickHouse token/boolean FTS or mMARCO may require more special cases.
  - Acceptance: A short design note identifies supported retrieval modes and required backend methods.

- [ ] Avoid hard-coding FTS as the opposite of vector search.
  - Evidence: current FTS mode sets `dim = 0`, Milvus infers `_is_fts` from `MilvusFtsConfig`, and schema creation is either dense-vector fields or text/BM25 sparse fields.
  - Impact: Future dense+BM25 hybrid search needs dense vector fields, text fields, BM25 sparse fields, and ranker config in the same workload. A binary vector-or-FTS model would force another refactor.
  - Acceptance: The first BM25 implementation uses an explicit workload kind, keeps hybrid out of scope, and avoids schema/config assumptions that prevent a future `HYBRID_DENSE_BM25` workload.

- [ ] Define backend capability boundaries for text search.
  - Evidence: target backends may expose different text-search capabilities, ranking defaults, analyzers, and scoring semantics.
  - Impact: Comparing results is only meaningful if the benchmark states what is normalized and what remains backend-specific.
  - Acceptance: Each backend config documents ranking model, analyzer/tokenizer defaults, tunable parameters, and unsupported options.

- [ ] Define FTS payload capability boundaries.
  - Evidence: Not every backend may support fetching stored text through the same API shape or with the same response cost.
  - Impact: Unsupported payload profiles should fail before a benchmark run, not after loading a large corpus.
  - Acceptance: Each supported FTS backend declares support for `ids_only` and `text` payload profiles or is gated out for unsupported profiles.

## Test Checklist

- [ ] Add unit tests for FTS dataset size limiting.
  - Acceptance: A fake ir-datasets corpus larger than the configured size only yields the configured number of documents.

- [ ] Add unit tests for MS MARCO qrels loading.
  - Acceptance: `load_ground_truth()` uses qrel relevance judgments and ignores scoreddoc candidates.

- [ ] Add unit tests for MS MARCO dataset translation.
  - Acceptance: The translator maps corpus documents, queries, and qrels into the shared FTS data model with stable IDs and qrel-preserving corpus caps.

- [ ] Add unit tests for HotpotQA dataset translation.
  - Acceptance: The translator maps corpus documents, queries, and qrels into the shared FTS data model with stable IDs.

- [ ] Add unit tests for FTS config generation.
  - Acceptance: UI-style config keys instantiate `MilvusFtsConfig` with the expected analyzer and BM25 fields.

- [ ] Add UI/task generation tests for the confirmed backend matrix.
  - Acceptance: Milvus, ElasticCloud, Vespa, and Turbopuffer can generate BM25 FTS tasks for MS MARCO and HotpotQA without a `KeyError`.

- [ ] Add tests for FTS payload profile propagation after PR 775 lands.
  - Acceptance: `ids_only` omits extra payload retrieval where supported; `text` passes a payload profile into serial and concurrent `search_documents()` calls; unsupported profiles fail fast; backends request the text field in `text` mode.

- [ ] Add dependency/import verification for the Docker requirements path.
  - Acceptance: A clean install from `install/requirements_py3.11.txt` can import modules that are imported at application startup.

## Suggested PR Language

Use wording like:

> This PR adds initial FTS support for Milvus BM25 sparse retrieval using MS MARCO. It benchmarks text-to-BM25 sparse-vector search as a separate mode from dense vector search. It does not yet cover generic FTS query semantics, hybrid dense+sparse retrieval, or non-Milvus backends.

After initial BM25 target coverage is implemented, use wording like:

> This feature benchmarks native BM25-ranked text retrieval across Milvus, ElasticCloud, Vespa, and Turbopuffer on English MS MARCO Passage and BEIR HotpotQA. It uses a shared corpus/query/qrels lifecycle with qrel-preserving corpus-size tiers and reports recall, NDCG, MRR, serial latency, and QPS using the existing VectorDBBench result schema. Backend-specific analyzer and ranking settings are documented per backend. ClickHouse token/boolean FTS, FTS text payload retrieval, and Chinese mMARCO are tracked as future follow-up work.
