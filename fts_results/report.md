# FTS E2E Results Master Report

This report compares the latest committed representative result for each backend and dataset configuration. Backend-specific historical reruns remain in each leaf `report.md`; this master view is for cross-backend comparison.

All local-server rows use the `r7i.4xlarge` server unless marked otherwise. TurboPuffer is a managed external backend, so its row is not directly comparable on server hardware. `Load s` is the VectorDBBench load duration. `QPS`, `p95 s`, and `p99 s` are from the search result in the raw JSON. Each row states its payload profile and concurrency list explicitly.

## Table of Contents

- [FTS Index And Ranking Configuration](#fts-index-and-ranking-configuration)
- [MS MARCO Small (100K documents)](#ms-marco-small-100k-documents)
- [MS MARCO Medium (1M documents)](#ms-marco-medium-1m-documents)
- [MS MARCO Large (8.8M documents)](#ms-marco-large-88m-documents)
- [HotpotQA Small (100K documents)](#hotpotqa-small-100k-documents)
- [HotpotQA Medium (1M documents)](#hotpotqa-medium-1m-documents)
- [HotpotQA Large (5.2M documents)](#hotpotqa-large-52m-documents)

## FTS Index And Ranking Configuration

The FTS runs did not use one uniform "backend default" configuration. Milvus and Vespa used explicit VDBBench FTS schemas/indexes. Elasticsearch used a plain text mapping with VDBBench index settings, then inherited Elasticsearch analyzer and BM25 defaults.

| Backend | Was Product Default? | Indexed Field | Index / Mapping | Analyzer / Linguistics | Ranking / Similarity | Explicit Args |
|---|---|---|---|---|---|---|
| Milvus | No. VDBBench explicitly configured the FTS sparse index and BM25 parameters. | Analyzer-enabled `text` field feeds generated `sparse_vector`. | `SPARSE_INVERTED_INDEX` on `sparse_vector`; BM25 function maps `text` to `sparse_vector`. | `standard` tokenizer, lowercase enabled, max token length `40`, stop words `null`. | `metric_type=BM25`; `bm25_k1=1.5`; `bm25_b=0.75`. | `inverted_index_algo=DAAT_MAXSCORE`; `drop_ratio_search=null`. |
| ElasticSearch | Partly. VDBBench explicitly configured mapping and index settings; analyzer and similarity were Elasticsearch defaults. | `text`. | Mapping: `doc_id: keyword`, `text: text`; standard Lucene inverted index for `text`. | Default `standard` analyzer: standard tokenizer, lowercase filter, stop filter disabled by default. | Default BM25 similarity: `k1=1.2`, `b=0.75`, `discount_overlaps=true`. | `number_of_shards=1`, `number_of_replicas=0`, `refresh_interval=30s`, force merge enabled. No HNSW/vector index settings were used for FTS. |
| Vespa | No. VDBBench explicitly deployed an FTS Vespa schema and rank profile. | `text`. | `text` is `string` with `index` and `summary`, plus `index: enable-bm25`; `id` is `summary` and `attribute`. | Vespa default string index text processing where not overridden, including tokenized text matching, normalization, and default stemming `best`. | Explicit rank profile `bm25` with first phase `bm25(text)`; BM25 parameters use Vespa defaults `k1=1.2`, `b=0.75`. | Query uses `userQuery()`, `ranking=bm25`, `default-index=text`, `type=any`; `VespaFtsConfig` has no extra index/search args, so raw JSON records `db_case_config={}`. |

Code references: Milvus FTS config is in `vectordb_bench/backend/clients/milvus/config.py` and index/search execution is in `vectordb_bench/backend/clients/milvus/milvus.py`. Elasticsearch FTS mapping/settings are in `vectordb_bench/backend/clients/elastic_cloud/config.py` and FTS search uses a `match` query in `vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py`. Vespa FTS schema and query construction are in `vectordb_bench/backend/clients/vespa/vespa.py`, with empty FTS case config in `vectordb_bench/backend/clients/vespa/config.py`.

Official docs used for product defaults: Milvus full text search, Elasticsearch standard analyzer and similarity docs, and Vespa BM25/schema/linguistics docs.

## MS MARCO Small (100K documents)

Text-payload rows used `payload_profile=text`, `k=100`, `concurrency_duration=30`, and explicit concurrency `1,10,20,40,60,80`. Local-server rows used the same `r7i.4xlarge` server host, recorded outside the repo in the local host config.

| Backend | Payload | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Milvus | ids_only | `r7i.4xlarge` | `Milvus/msmarco/raw_results/result_20260602_fts-e2e-milvus-msmarco-small-r7i_milvus.json` | 230.3305 | 9359.8351 | 0.9157 | 0.7157 | 0.6653 | 0.0026 | 0.0029 | 1/5/10/20 | 528.3714 / 3129.5306 / 5750.1304 / 9359.8351 |
| Milvus | text | `r7i.4xlarge` | `Milvus/msmarco/raw_results/result_20260603_fts-e2e-milvus-msmarco-small-text-r7i-rerun_milvus.json` | 230.4392 | 9569.0676 | 0.9157 | 0.7157 | 0.6653 | 0.0029 | 0.0032 | 1/10/20/40/60/80 | 468.2255 / 4857.3898 / 8011.7230 / 9279.3577 / 9569.0676 / 9266.8844 |
| ElasticSearch | ids_only | `r7i.4xlarge` | `ElasticSearch/msmarco/raw_results/result_20260602_fts-e2e-elastic-msmarco-small-r7i_elasticcloud.json` | 59.4276 | 8689.3499 | 0.9118 | 0.7159 | 0.6665 | 0.0030 | 0.0035 | 1/5/10/20 | 396.5015 / 2534.0129 / 5536.5659 / 8689.3499 |
| ElasticSearch | text | `r7i.4xlarge` | `ElasticSearch/msmarco/raw_results/result_20260603_fts-e2e-elastic-msmarco-small-text-r7i_elasticcloud.json` | 57.8052 | 4177.1357 | 0.9118 | 0.7159 | 0.6665 | 0.0046 | 0.0051 | 1/10/20/40/60/80 | 242.0833 / 2599.9459 / 3941.7042 / 4158.8964 / 4177.1357 / 4155.8592 |
| Vespa | ids_only | `r7i.4xlarge` | `Vespa/msmarco/raw_results/result_20260602_fts-e2e-vespa-msmarco-small-r7i_vespa.json` | 79.2473 | 734.5241 | 0.9416 | 0.7509 | 0.7015 | 0.0184 | 0.0230 | 1/5/10/20 | 91.7244 / 512.5352 / 347.9730 / 734.5241 |
| Vespa | text | `r7i.4xlarge` | `Vespa/msmarco/raw_results/result_20260603_fts-e2e-vespa-msmarco-small-text-r7i_vespa.json` | 78.8999 | 788.0555 | 0.9416 | 0.7509 | 0.7015 | 0.0193 | 0.0236 | 1/10/20/40/60/80 | 64.6508 / 786.3064 / 131.0499 / 788.0555 / 422.9538 / 365.3142 |
| TurboPuffer | ids_only | managed backend | `Turbopuffer/msmarco/raw_results/result_20260601_fts-e2e-tpuf-msmarco-small_turbopuffer.json` | 290.5625 | 257.3771 | 0.9125 | 0.7156 | 0.6659 | 0.0840 | 0.1081 | 1/5/10/20 | 1.3357 / 49.4967 / 126.8548 / 257.3771 |

## MS MARCO Medium (1M documents)

Rows below are the 2026-06-04 six-concurrency rerun using explicit concurrency `1,10,20,40,60,80`. Older ids-only `1,5,10,20` baselines remain in the backend-specific reports for stability comparison.

| Backend | Payload | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Milvus | ids_only | `r7i.4xlarge` | `Milvus/msmarco/raw_results/result_20260604_fts-msmarco-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T041648Z_milvus.json` | 2048.1231 | 5139.5920 | 0.8048 | 0.5174 | 0.4458 | 0.0053 | 0.0071 | 1/10/20/40/60/80 | 433.1311 / 2976.3670 / 3973.3733 / 4750.5123 / 5053.7822 / 5139.5920 |
| Milvus | text | `r7i.4xlarge` | `Milvus/msmarco/raw_results/result_20260604_fts-msmarco-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T041648Z_milvus.json` | 2048.2360 | 4677.4078 | 0.8048 | 0.5174 | 0.4458 | 0.0057 | 0.0075 | 1/10/20/40/60/80 | 378.3115 / 2732.7863 / 3656.8277 / 4353.0352 / 4602.9011 / 4677.4078 |
| ElasticSearch | ids_only | `r7i.4xlarge` | `ElasticSearch/msmarco/raw_results/result_20260604_fts-msmarco-medium-elastic-ids-c1-10-20-40-60-80-r7i-20260604T041648Z_elasticcloud.json` | 140.1544 | 4473.8674 | 0.8028 | 0.5222 | 0.4526 | 0.0063 | 0.0086 | 1/10/20/40/60/80 | 260.6360 / 2883.9739 / 4166.7860 / 4405.5505 / 4473.8674 / 4458.2345 |
| ElasticSearch | text | `r7i.4xlarge` | `ElasticSearch/msmarco/raw_results/result_20260604_fts-msmarco-medium-elastic-text-c1-10-20-40-60-80-r7i-20260604T041648Z_elasticcloud.json` | 139.6663 | 2696.5048 | 0.8028 | 0.5222 | 0.4526 | 0.0079 | 0.0101 | 1/10/20/40/60/80 | 178.1539 / 1787.4005 / 2605.4203 / 2688.4985 / 2680.9282 / 2696.5048 |
| Vespa | ids_only | `r7i.4xlarge` | `Vespa/msmarco/raw_results/result_20260604_fts-msmarco-medium-vespa-ids-c1-10-20-40-60-80-r7i-20260604T041648Z_vespa.json` | 581.5774 | 257.0647 | 0.8409 | 0.5499 | 0.4767 | 0.1231 | 0.1688 | 1/10/20/40/60/80 | 17.2619 / 153.1753 / 217.6157 / 230.2074 / 238.9556 / 257.0647 |
| Vespa | text | `r7i.4xlarge` | `Vespa/msmarco/raw_results/result_20260604_fts-msmarco-medium-vespa-text-c1-10-20-40-60-80-r7i-20260604T041648Z_vespa.json` | 581.4244 | 251.4636 | 0.8409 | 0.5499 | 0.4767 | 0.1248 | 0.1702 | 1/10/20/40/60/80 | 15.0937 / 133.4407 / 199.3716 / 209.5499 / 234.2022 / 251.4636 |

## MS MARCO Large (8.8M documents)

No committed raw result yet.

## HotpotQA Small (100K documents)

No committed raw result yet.

## HotpotQA Medium (1M documents)

Rows below are the 2026-06-04 six-concurrency rerun using explicit concurrency `1,10,20,40,60,80`. Older ids-only `1,5,10,20` baselines and the previous failed Vespa text-payload attempt remain in the backend-specific reports.

| Backend | Payload | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Milvus | ids_only | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260604_fts-hotpotqa-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T074646Z_milvus.json` | 2040.9336 | 1865.4681 | 0.8378 | 0.7246 | 0.8561 | 0.0122 | 0.0170 | 1/10/20/40/60/80 | 255.0087 / 1364.3522 / 1378.9975 / 1702.7745 / 1851.3955 / 1865.4681 |
| Milvus | text | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260604_fts-hotpotqa-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T074646Z_milvus.json` | 2033.2594 | 1714.0357 | 0.8378 | 0.7246 | 0.8561 | 0.0124 | 0.0170 | 1/10/20/40/60/80 | 223.9828 / 1224.1637 / 1558.9074 / 1669.1467 / 1687.2785 / 1714.0357 |
| ElasticSearch | ids_only | `r7i.4xlarge` | `ElasticSearch/hotpotqa/raw_results/result_20260604_fts-hotpotqa-medium-elastic-ids-c1-10-20-40-60-80-r7i-20260604T074646Z_elasticcloud.json` | 139.2256 | 1581.7165 | 0.8378 | 0.7287 | 0.8598 | 0.0150 | 0.0212 | 1/10/20/40/60/80 | 119.3122 / 1106.0982 / 1552.4142 / 1581.7165 / 1574.0491 / 1579.6451 |
| ElasticSearch | text | `r7i.4xlarge` | `ElasticSearch/hotpotqa/raw_results/result_20260604_fts-hotpotqa-medium-elastic-text-c1-10-20-40-60-80-r7i-20260604T074646Z_elasticcloud.json` | 140.1574 | 1238.7840 | 0.8378 | 0.7287 | 0.8598 | 0.0171 | 0.0237 | 1/10/20/40/60/80 | 94.8516 / 870.2786 / 1203.6183 / 1230.3996 / 1238.7840 / 1229.2721 |
| Vespa | ids_only | `r7i.4xlarge` | `Vespa/hotpotqa/raw_results/result_20260604_fts-hotpotqa-medium-vespa-ids-c1-10-20-40-60-80-r7i-20260604T074646Z_vespa.json` | 579.2518 | 181.5240 | 0.8309 | 0.7208 | 0.8500 | 0.2628 | 0.3223 | 1/10/20/40/60/80 | 6.6652 / 58.1128 / 81.3796 / 104.5544 / 140.9124 / 181.5240 |
| Vespa | text | `r7i.4xlarge` | `Vespa/hotpotqa/raw_results/result_20260604_fts-hotpotqa-medium-vespa-text-c1-10-20-40-60-80-r7i-20260604T074646Z_vespa.json` | 579.3951 | 177.2947 | 0.8309 | 0.7208 | 0.8500 | 0.2683 | 0.3313 | 1/10/20/40/60/80 | 5.2029 / 55.8888 / 79.5598 / 100.9787 / 138.3337 / 177.2947 |

Vespa text completed in the 2026-06-04 rerun, but emitted backend timeout warnings during concurrency 60 and 80. The raw JSON is valid and included.

## HotpotQA Large (5.2M documents)

Matrix rows used explicit concurrency `20,40,80`, `k=100`, and `concurrency_duration=30`. Payload `ids_only` returns ids only; payload `text` returns ids plus text payload.

| Backend | Payload | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ElasticSearch | ids_only | `r7i.4xlarge` | `ElasticSearch/hotpotqa/raw_results/result_20260602_fts-e2e-elastic-hotpotqa-large-r7i_elasticcloud.json` | 550.6164 | 476.2610 | 0.7637 | 0.6243 | 0.7549 | 0.0503 | 0.0755 | 1/5/10/20 | 41.0129 / 202.7703 / 356.3845 / 476.2610 |
| ElasticSearch | text | `r7i.4xlarge` | `ElasticSearch/hotpotqa/raw_results/result_20260604_fts-matrix-elastic-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z_elasticcloud.json` | 554.4492 | 435.1027 | 0.7637 | 0.6243 | 0.7549 | 0.0518 | 0.0766 | 20/40/80 | 402.3090 / 435.1027 / 434.3993 |
| Milvus | ids_only | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260602_fts-e2e-milvus-hotpotqa-large-r7i_milvus.json` | 10583.8485 | 394.4417 | 0.7573 | 0.6129 | 0.7410 | 0.0212 | 0.0299 | 1/5/10/20 | 88.1695 / 336.8579 / 388.2553 / 394.4417 |
| Milvus | ids_only | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260603_fts-matrix-milvus-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z_milvus.json` | 10583.8402 | 411.7323 | 0.7573 | 0.6129 | 0.7410 | 0.0211 | 0.0305 | 20/40/80 | 400.7550 / 407.1847 / 411.7323 |
| Milvus | text | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260603_fts-matrix-milvus-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z_milvus.json` | 10583.7873 | 409.4366 | 0.7573 | 0.6129 | 0.7410 | 0.0214 | 0.0308 | 20/40/80 | 395.3148 / 407.5527 / 409.4366 |
| Vespa | ids_only | `r7i.4xlarge` | `Vespa/hotpotqa/raw_results/result_20260602_fts-e2e-vespa-hotpotqa-large-r7i_vespa.json` | 2954.2589 | 46.3472 | 0.6754 | 0.5460 | 0.6640 | 0.4460 | 0.4465 | 1/5/10/20 | 3.3531 / 13.1559 / 24.4787 / 46.3472 |

Excluded milestone run: `elastic/hotpotqa-large/ids` with concurrency `20,40,80` loaded successfully (`545.7043s`) and completed concurrency 20/40 (`447.5993 / 480.0184 QPS`), then hung after starting concurrency 80. VDBBench emitted only a zero-metric placeholder JSON, so no raw result is committed or compared.
