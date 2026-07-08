# Session Progress

**Date:** 2026-05-26
**Working Directory:** /home/ubuntu/VectorDBBench

## Completed Work

### 1. Reviewed Current FTS Implementation
Confirmed that existing FTS support is Milvus BM25-only and implemented through a mostly separate path from the vector benchmark lifecycle. Captured correctness gaps including qrels handling, ID types, dataset size caps, dependency wiring, and UI/backend config mismatch.
- Files: `docs/fts-feature-review-checklist.md` (created/modified), `vectordb_bench/backend/dataset.py` (read), `vectordb_bench/backend/cases.py` (read), `vectordb_bench/backend/task_runner.py` (read), `vectordb_bench/backend/runner/serial_runner.py` (read), `vectordb_bench/backend/clients/milvus/milvus.py` (read), `vectordb_bench/backend/clients/milvus/config.py` (read), `vectordb_bench/frontend/config/dbCaseConfigs.py` (read), `vectordb_bench/cli/vectordbbench.py` (read)

### 2. Defined Dataset And BM25 Scope
Documented the initial dataset matrix as English MS MARCO Passage and BEIR HotpotQA with Small, Medium, and Large corpus-size tiers. Confirmed first draft scope as native BM25-ranked top-k retrieval across Milvus, Elasticsearch, Vespa, and Turbopuffer, with ClickHouse and Chinese mMARCO deferred.
- Files: `docs/superpowers/specs/2026-05-26-fts-dataset-options-design.md` (created/modified), `docs/fts-feature-review-checklist.md` (modified)

### 3. Defined FTS Implementation Architecture
Documented the agreed split: keep FTS dataset/backend/metric semantics separate while reusing the shared benchmark lifecycle, task stages, search runners, result schema, UI/CLI flow, and capability gating where practical. Captured the recommendation to avoid broad retrieval refactors before the BM25 backend set is implemented.
- Files: `docs/superpowers/specs/2026-05-26-fts-implementation-overall-design.md` (created), `docs/fts-feature-review-checklist.md` (read)

### 4. Reviewed Payload Design From PR 775
Inspected PR 775's cloud payload profile pattern and documented that FTS should support `ids_only` and opt-in `text` response payload profiles without creating separate FTS payload case classes. The design now says to reuse PR 775 payload infrastructure when available and keep metrics ID-based while measuring payload response cost.
- Files: `docs/superpowers/specs/2026-05-26-fts-implementation-overall-design.md` (modified), `docs/fts-feature-review-checklist.md` (modified), `vectordb_bench/backend/payload.py` from `origin/pr/775` (read), `vectordb_bench/backend/cases.py` from `origin/pr/775` (read), `vectordb_bench/backend/runner/serial_runner.py` from `origin/pr/775` (read), `vectordb_bench/backend/runner/mp_runner.py` from `origin/pr/775` (read), `vectordb_bench/metric.py` from `origin/pr/775` (read), `tests/test_cloud_payload_case.py` from `origin/pr/775` (read), `tests/test_cloud_payload_search.py` from `origin/pr/775` (read)

### 5. Discussed Backend API And Hybrid Future-Proofing
Agreed that FTS should use separate optional semantic methods such as `insert_documents()` and `search_documents()` rather than overloading `insert_embeddings()` and `search_embedding()`. Confirmed the current binary vector-or-FTS design would make future hybrid search awkward, and identified `WorkloadKind` plus composable dense/text/hybrid configs as the future-proofing boundary.
- Files: `vectordb_bench/backend/clients/api.py` (read), `vectordb_bench/backend/clients/milvus/milvus.py` (read), `vectordb_bench/backend/clients/milvus/config.py` (read), `vectordb_bench/backend/task_runner.py` (read)

### 6. Finalized Remaining FTS Design Choices
Recorded that the first Elasticsearch-compatible target is `DB.ElasticCloud`, first-pass FTS should run IDs-only and defer text payload retrieval until PR 775's shared payload machinery is available, and Large dataset tiers should be advanced/opt-in. Updated the design docs to include these decisions and the hybrid future-proofing constraint.
- Files: `docs/fts-feature-review-checklist.md` (modified), `docs/superpowers/specs/2026-05-26-fts-dataset-options-design.md` (modified), `docs/superpowers/specs/2026-05-26-fts-implementation-overall-design.md` (modified)

## Current Status
The FTS design has enough agreed scope to move into an implementation plan. The latest design edits are not committed yet.

## Open Issues
- If FTS filter cases are added later, decide whether their ground truth should use random filtering or GT-preserving filtering. Random filtering is simpler but can remove judged-relevant docs from low-selectivity cases; GT-preserving filtering keeps positive qrel docs inside the filtered subset and then fills the subset to the target filter rate, but requires validating that the filtered bucket can contain all required qrel docs.
