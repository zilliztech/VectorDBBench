# FTS Dataset Options Design

**Date:** 2026-05-26
**Status:** Draft for review
**Scope:** Dataset choices and size-tier policy for full-text search benchmarks in VectorDBBench.

## Decision

Use English MS MARCO Passage and BEIR HotpotQA as the initial FTS benchmark datasets. Do not expose `nano-beir/hotpotqa` in the first public benchmark matrix. Track Chinese mMARCO as a future multilingual analyzer/tokenizer stress dataset, not as an initial required dataset.

The benchmark should expose dataset sizes as corpus-size tiers, similar to existing VectorDBBench vector datasets:

- `Small`: 100K documents
- `Medium`: 1M documents
- `Large`: full native corpus

For capped tiers, use a qrel-preserving corpus cap. The cap must include all documents with positive qrels for the selected query split, then fill remaining slots deterministically from the native corpus order. If the number of qrel-required documents exceeds the cap, the dataset configuration is invalid.

## Initial Dataset Matrix

| Dataset option | ir-datasets source | Query split | Corpus size |
| --- | --- | --- | --- |
| `MS MARCO Small (100K documents)` | `msmarco-passage/dev/small` | 6,980 queries, 7,437 qrels | 100K cap |
| `MS MARCO Medium (1M documents)` | `msmarco-passage/dev/small` | 6,980 queries, 7,437 qrels | 1M cap |
| `MS MARCO Large (8.8M documents)` | `msmarco-passage/dev/small` | 6,980 queries, 7,437 qrels | Full 8,841,823 docs |
| `HotpotQA Small (100K documents)` | `beir/hotpotqa/test` | 7,405 queries, 14,810 qrels | 100K cap |
| `HotpotQA Medium (1M documents)` | `beir/hotpotqa/test` | 7,405 queries, 14,810 qrels | 1M cap |
| `HotpotQA Large (5.2M documents)` | `beir/hotpotqa/test` | 7,405 queries, 14,810 qrels | Full 5,233,329 docs |

Sources:

- MS MARCO Passage: https://ir-datasets.com/msmarco-passage.html
- HotpotQA via BEIR: https://ir-datasets.com/beir.html#beir/hotpotqa
- mMARCO, future multilingual candidate: https://ir-datasets.com/mmarco.html

## Rationale

Native ir-datasets split names do not map cleanly to benchmark size. For example, `msmarco-passage/dev/small` and mMARCO `dev/small` still inherit the full 8.8M-document corpus. Labeling those as "small" without a corpus cap makes load time, storage footprint, index build time, and QPS incomparable with VectorDBBench's existing `100K`, `1M`, and `10M` style options.

Qrel-preserving caps avoid the main correctness problem with simple first-N caps. If a relevant document is outside the capped corpus, recall, NDCG, and MRR become artificially impossible for that query. Including all positive-qrel documents first keeps quality metrics meaningful while still allowing predictable benchmark sizes.

MS MARCO English should be the first baseline because most target backends can run comparable English BM25-style text retrieval with less analyzer normalization risk. HotpotQA adds a different retrieval workload and a large BEIR corpus without introducing multilingual tokenization as the first problem.

Chinese mMARCO is still valuable, but it should come later. Chinese text has no whitespace-delimited words, so tokenizer and analyzer choices can dominate quality metrics. That makes it useful for a dedicated multilingual/analyzer benchmark, but noisy for the initial cross-backend baseline.

## Dataset Adapter Requirements

- Load corpus, queries, and qrels through ir-datasets adapters.
- Treat query IDs and document IDs as strings end-to-end.
- Build ground truth from `qrels_iter()` with positive relevance only.
- Do not use `scoreddocs_iter()` as relevance ground truth.
- Keep dataset adapters separate from vector parquet datasets, but use the shared retrieval benchmark lifecycle where practical.
- Preserve exact dataset option labels in result metadata so backend x dataset comparisons are unambiguous.

## Cap Construction

For `Small` and `Medium`:

1. Load qrels for the selected query split.
2. Build a deterministic ordered set of positive-qrel document IDs.
3. Validate that `len(required_doc_ids) <= target_size`.
4. Include all required documents in the benchmark corpus.
5. Stream native documents in ir-datasets order and append non-required documents until the corpus reaches `target_size`.
6. Keep the same selected query and qrels split across size tiers unless a future design explicitly defines query-count tiers.

For `Large`, stream the full native corpus.

## Follow-Up Decisions

- Whether to add an evaluation-heavy MS MARCO variant using `msmarco-passage/dev` instead of `dev/small`.
- Whether to add Chinese mMARCO, likely `mmarco/v2/zh/dev/small`, as a separate multilingual/analyzer benchmark.
- Whether each backend should expose configurable analyzers or only documented defaults for the initial benchmark.
