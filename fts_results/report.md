# FTS E2E Results Master Report

This report compares the latest committed representative result for each backend and dataset configuration. Backend-specific historical reruns remain in each leaf `report.md`; this master view is for cross-backend comparison.

All local-server rows use the `r7i.4xlarge` server unless marked otherwise. TurboPuffer is a managed external backend, so its row is not directly comparable on server hardware. `Load s` is the VectorDBBench load duration. `QPS`, `p95 s`, and `p99 s` are from the search result in the raw JSON. Concurrent QPS is shown at concurrency `1 / 5 / 10 / 20`.

## MS MARCO Small (100K documents)

| Backend | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS 1/5/10/20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Milvus | `r7i.4xlarge` | `Milvus/msmarco/raw_results/result_20260602_fts-e2e-milvus-msmarco-small-r7i_milvus.json` | 230.3305 | 9359.8351 | 0.9157 | 0.7157 | 0.6653 | 0.0026 | 0.0029 | 528.3714 / 3129.5306 / 5750.1304 / 9359.8351 |
| ElasticSearch | `r7i.4xlarge` | `ElasticSearch/msmarco/raw_results/result_20260602_fts-e2e-elastic-msmarco-small-r7i_elasticcloud.json` | 59.4276 | 8689.3499 | 0.9118 | 0.7159 | 0.6665 | 0.0030 | 0.0035 | 396.5015 / 2534.0129 / 5536.5659 / 8689.3499 |
| Vespa | `r7i.4xlarge` | `Vespa/msmarco/raw_results/result_20260602_fts-e2e-vespa-msmarco-small-r7i_vespa.json` | 79.2473 | 734.5241 | 0.9416 | 0.7509 | 0.7015 | 0.0184 | 0.0230 | 91.7244 / 512.5352 / 347.9730 / 734.5241 |
| TurboPuffer | managed backend | `Turbopuffer/msmarco/raw_results/result_20260601_fts-e2e-tpuf-msmarco-small_turbopuffer.json` | 290.5625 | 257.3771 | 0.9125 | 0.7156 | 0.6659 | 0.0840 | 0.1081 | 1.3357 / 49.4967 / 126.8548 / 257.3771 |

## MS MARCO Medium (1M documents)

| Backend | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS 1/5/10/20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ElasticSearch | `r7i.4xlarge` | `ElasticSearch/msmarco/raw_results/result_20260602_fts-e2e-elastic-msmarco-medium-r7i_elasticcloud.json` | 148.5277 | 3986.3734 | 0.8028 | 0.5222 | 0.4526 | 0.0065 | 0.0089 | 240.5419 / 1348.5987 / 2719.8093 / 3986.3734 |
| Milvus | `r7i.4xlarge` | `Milvus/msmarco/raw_results/result_20260602_fts-e2e-milvus-msmarco-medium-r7i_milvus.json` | 2042.1265 | 3857.7069 | 0.8048 | 0.5174 | 0.4458 | 0.0056 | 0.0074 | 284.4606 / 1608.7846 / 2858.1176 / 3857.7069 |
| Vespa | `r7i.4xlarge` | `Vespa/msmarco/raw_results/result_20260602_fts-e2e-vespa-msmarco-medium-r7i_vespa.json` | 585.7496 | 196.8213 | 0.8409 | 0.5499 | 0.4767 | 0.1268 | 0.1744 | 17.3137 / 85.2530 / 139.2205 / 196.8213 |

## HotpotQA Medium (1M documents)

| Backend | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS 1/5/10/20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Milvus | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260602_fts-e2e-milvus-hotpotqa-medium-r7i_milvus.json` | 2031.2796 | 1596.6340 | 0.8378 | 0.7246 | 0.8561 | 0.0123 | 0.0170 | 146.0629 / 726.2209 / 1273.3380 / 1596.6340 |
| ElasticSearch | `r7i.4xlarge` | `ElasticSearch/hotpotqa/raw_results/result_20260602_fts-e2e-elastic-hotpotqa-medium-r7i_elasticcloud.json` | 142.0589 | 1410.3787 | 0.8378 | 0.7287 | 0.8598 | 0.0159 | 0.0224 | 111.5252 / 558.2304 / 1034.1176 / 1410.3787 |
| Vespa | `r7i.4xlarge` | `Vespa/hotpotqa/raw_results/result_20260602_fts-e2e-vespa-hotpotqa-medium-r7i_vespa.json` | 575.5304 | 80.0872 | 0.8309 | 0.7208 | 0.8500 | 0.2647 | 0.3261 | 6.4907 / 33.8533 / 57.9060 / 80.0872 |

## HotpotQA Small (100K documents)

No committed raw result yet.

## MS MARCO Large (8.8M documents)

No committed raw result yet.

## HotpotQA Large (5.2M documents)

| Backend | Context | Raw JSON | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS 1/5/10/20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ElasticSearch | `r7i.4xlarge` | `ElasticSearch/hotpotqa/raw_results/result_20260602_fts-e2e-elastic-hotpotqa-large-r7i_elasticcloud.json` | 550.6164 | 476.2610 | 0.7637 | 0.6243 | 0.7549 | 0.0503 | 0.0755 | 41.0129 / 202.7703 / 356.3845 / 476.2610 |
| Milvus | `r7i.4xlarge` | `Milvus/hotpotqa/raw_results/result_20260602_fts-e2e-milvus-hotpotqa-large-r7i_milvus.json` | 10583.8485 | 394.4417 | 0.7573 | 0.6129 | 0.7410 | 0.0212 | 0.0299 | 88.1695 / 336.8579 / 388.2553 / 394.4417 |
| Vespa | `r7i.4xlarge` | `Vespa/hotpotqa/raw_results/result_20260602_fts-e2e-vespa-hotpotqa-large-r7i_vespa.json` | 2954.2589 | 46.3472 | 0.6754 | 0.5460 | 0.6640 | 0.4460 | 0.4465 | 3.3531 / 13.1559 / 24.4787 / 46.3472 |
