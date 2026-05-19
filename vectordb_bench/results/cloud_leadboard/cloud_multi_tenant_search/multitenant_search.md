# Multitenant Search Results

## Purpose

This document summarizes the completed CloudMultiTenantSearchCase run for cloud vector database products. The run measures concurrent QPS and latency for 1,000-tenant Cohere 10M search while varying filter mode and returned payload profile.

## Run Summary

| Item | Value |
|---|---|
| Dataset | Large Cohere (768dim, 10M) |
| Metric | cosine |
| TopK | 50 |
| Tenant count | 1,000 |
| Tenant label format | `tenant_0000` through `tenant_0999` |
| Tenant assignment | `tenant_id = row_id % tenant_count` |
| Search phase | concurrent QPS and latency only |
| Concurrency list | `60,80` for Zilliz/Turbopuffer; `4` for Pinecone c4-only detached run |
| Duration | 30s per concurrency |
| Framework commit | `df1521796acd63d62042e182e7addad02c21e792` |
| Framework branch | `cloud-payload-search-case` |
| Raw result manifest | [raw_results/manifest.jsonl](raw_results/manifest.jsonl) |

## Filter Rate Semantics

Integer filtered rows use the actual `--cloud-filter-rate` value. The expression is `id >= int(dataset_size * filter_rate)`, so a 99.9% filter rate leaves roughly 0.1% of rows as candidates. Scalar label filtered rows use `--cloud-label-percentage`; a 1% scalar-label filter rate means `label == "label_1p"`.

| Filter type | Rates used |
|---|---|
| Integer filter | 99.9%, 99%, 90%, 50% |
| Scalar label filter | 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50% |

## Product Status

| Product | Completed cases | Status |
|---|---:|---|
| Zilliz Cloud Tiered 1CU | 42 | measured |
| Zilliz Cloud Capacity 2CU | 42 | measured |
| Turbopuffer | 42 | measured |
| Pinecone Serverless | 28 | measured c4-only; IDs-only and vector payloads only |

## Notes

- Zilliz Cloud Tiered 1CU scalar-label payload QPS was higher than IDs-only for several scalar-label filter rates. The harness paths were checked and the payload branches are not reversed, but these rows should be rebenched before drawing a product conclusion from that inversion.
- Vector payload returns the 768D vector field and is expected to be substantially heavier than IDs-only or scalar-label payloads.
- Pinecone Serverless results were collected separately with concurrency `4`; keep them separate from the `60,80` comparison rows when interpreting leaderboard results.

## Zilliz Cloud Tiered 1CU

| Item | Value |
|---|---|
| Product key | `zilliz_cloud_tiered_1cu` |
| Notes | Zilliz Cloud tiered cluster configured as 1 CU for multitenant search. |

### Unfiltered Search

| Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---|---:|---:|---:|---|---|---|---:|---|
| IDs only | 473.6019 | 481.1791 | [481.1791](raw_results/zilliz_cloud_tiered_1cu/unfiltered/na/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_unfiltered_na_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1246s / 0.1620s | 0.1430s / 0.1952s | 0.1942s / 0.2171s | 1,000 | measured |
| scalar label | 444.9385 | 431.0545 | [444.9385](raw_results/zilliz_cloud_tiered_1cu/unfiltered/na/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_unfiltered_na_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1325s / 0.1808s | 0.1496s / 0.2217s | 0.1805s / 0.2591s | 1,800 | measured |
| vector | 31.4324 | 34.4462 | [34.4462](raw_results/zilliz_cloud_tiered_1cu/unfiltered/na/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_unfiltered_na_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 1.8547s / 2.2449s | 4.0452s / 2.7293s | 4.4013s / 2.7684s | 154,600 | measured |

### Integer Filtered Search

| Filter rate | Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 99.9% | IDs only | 944.0877 | 852.4406 | [944.0877](raw_results/zilliz_cloud_tiered_1cu/int_filter/99_9p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_0_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0624s / 0.0912s | 0.0835s / 0.1300s | 0.1299s / 0.1734s | 1,000 | measured |
| 99.9% | scalar label | 918.9192 | 937.8986 | [937.8986](raw_results/zilliz_cloud_tiered_1cu/int_filter/99_9p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_0_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0641s / 0.0829s | 0.0789s / 0.1147s | 0.1367s / 0.1718s | 1,800 | measured |
| 99.9% | vector | 624.7683 | 609.8326 | [624.7683](raw_results/zilliz_cloud_tiered_1cu/int_filter/99_9p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_int_filter_0_1p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.0942s / 0.1275s | 0.1191s / 0.1574s | 0.1291s / 0.1709s | 154,600 | measured |
| 99% | IDs only | 845.7500 | 857.2543 | [857.2543](raw_results/zilliz_cloud_tiered_1cu/int_filter/99p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0695s / 0.0907s | 0.0918s / 0.1258s | 0.1361s / 0.1810s | 1,000 | measured |
| 99% | scalar label | 808.5485 | 780.1259 | [808.5485](raw_results/zilliz_cloud_tiered_1cu/int_filter/99p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0728s / 0.0997s | 0.0921s / 0.1331s | 0.1361s / 0.1801s | 1,800 | measured |
| 99% | vector | 456.5656 | 465.9550 | [465.9550](raw_results/zilliz_cloud_tiered_1cu/int_filter/99p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_int_filter_1p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.1293s / 0.1668s | 0.1566s / 0.1985s | 0.2595s / 0.2164s | 154,600 | measured |
| 90% | IDs only | 629.3629 | 711.7052 | [711.7052](raw_results/zilliz_cloud_tiered_1cu/int_filter/90p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_10p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0937s / 0.1093s | 0.1308s / 0.1446s | 0.1812s / 0.1854s | 1,000 | measured |
| 90% | scalar label | 689.2834 | 691.9848 | [691.9848](raw_results/zilliz_cloud_tiered_1cu/int_filter/90p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_10p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0855s / 0.1126s | 0.1099s / 0.1498s | 0.1344s / 0.1708s | 1,800 | measured |
| 90% | vector | 293.1843 | 411.9756 | [411.9756](raw_results/zilliz_cloud_tiered_1cu/int_filter/90p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_int_filter_10p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.2009s / 0.1889s | 0.4769s / 0.2151s | 0.9172s / 0.2734s | 154,600 | measured |
| 50% | IDs only | 282.2915 | 470.6729 | [470.6729](raw_results/zilliz_cloud_tiered_1cu/int_filter/50p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_50p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.2086s / 0.1657s | 0.4477s / 0.2035s | 1.4360s / 0.2526s | 1,000 | measured |
| 50% | scalar label | 450.5191 | 462.5285 | [462.5285](raw_results/zilliz_cloud_tiered_1cu/int_filter/50p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_int_filter_50p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1310s / 0.1682s | 0.1626s / 0.2064s | 0.1867s / 0.2350s | 1,800 | measured |
| 50% | vector | 39.9734 | 56.4443 | [56.4443](raw_results/zilliz_cloud_tiered_1cu/int_filter/50p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_int_filter_50p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 1.4754s / 1.3814s | 1.9572s / 1.6589s | 2.5759s / 2.1692s | 154,600 | measured |

### Scalar Label Filtered Search

| Filter rate | Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 0.1% | IDs only | 379.0223 | 387.8166 | [387.8166](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_0_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1557s / 0.2007s | 0.1888s / 0.2441s | 0.2273s / 0.2605s | 1,000 | measured |
| 0.1% | scalar label | 384.4926 | 383.2965 | [384.4926](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_0_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1534s / 0.2030s | 0.1811s / 0.2471s | 0.1967s / 0.3035s | 1,800 | measured |
| 0.1% | vector | 293.3546 | 326.8364 | [326.8364](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_0_1p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.2015s / 0.2384s | 0.2787s / 0.2646s | 0.3541s / 0.2896s | 154,600 | measured |
| 0.2% | IDs only | 364.1257 | 400.6982 | [400.6982](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_0_2p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1620s / 0.1944s | 0.2017s / 0.2278s | 0.2259s / 0.2536s | 1,000 | measured |
| 0.2% | scalar label | 381.3891 | 387.0038 | [387.0038](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_0_2p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1546s / 0.2013s | 0.1815s / 0.2470s | 0.2041s / 0.2696s | 1,800 | measured |
| 0.2% | vector | 249.8844 | 306.5062 | [306.5062](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_0_2p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.2362s / 0.2538s | 0.4050s / 0.2853s | 0.4676s / 0.3019s | 154,600 | measured |
| 0.5% | IDs only | 321.5244 | 365.6331 | [365.6331](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_0_5p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1840s / 0.2129s | 0.2788s / 0.2581s | 0.3832s / 0.2754s | 1,000 | measured |
| 0.5% | scalar label | 364.7357 | 368.4364 | [368.4364](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_0_5p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1615s / 0.2115s | 0.1890s / 0.2502s | 0.2215s / 0.2746s | 1,800 | measured |
| 0.5% | vector | 148.7183 | 265.8323 | [265.8323](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_0_5p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.3965s / 0.2933s | 0.7820s / 0.3403s | 0.9525s / 0.3660s | 154,600 | measured |
| 1% | IDs only | 289.4898 | 374.6763 | [374.6763](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.2038s / 0.2077s | 0.3928s / 0.2469s | 0.5805s / 0.2847s | 1,000 | measured |
| 1% | scalar label | 364.8097 | 350.1425 | [364.8097](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1617s / 0.2225s | 0.1920s / 0.2670s | 0.2939s / 0.3362s | 1,800 | measured |
| 1% | vector | 70.5499 | 152.0617 | [152.0617](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/1p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_1p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 0.8387s / 0.5125s | 1.2743s / 0.7973s | 1.6661s / 1.0396s | 154,600 | measured |
| 2% | IDs only | 130.0559 | 353.2406 | [353.2406](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_2p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.4532s / 0.2205s | 1.1587s / 0.2647s | 1.2666s / 0.3238s | 1,000 | measured |
| 2% | scalar label | 348.2551 | 349.6334 | [349.6334](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_2p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1698s / 0.2229s | 0.2013s / 0.2700s | 0.2183s / 0.3328s | 1,800 | measured |
| 2% | vector | 47.0214 | 65.1543 | [65.1543](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/2p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_2p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 1.2590s / 1.2042s | 1.7567s / 1.4523s | 2.1241s / 1.8916s | 154,600 | measured |
| 5% | IDs only | 67.5627 | 223.5542 | [223.5542](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_5p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.8737s / 0.3490s | 1.7081s / 0.5422s | 1.9779s / 0.6266s | 1,000 | measured |
| 5% | scalar label | 292.7097 | 315.3831 | [315.3831](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_5p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.2014s / 0.2469s | 0.2393s / 0.2906s | 0.2960s / 0.3271s | 1,800 | measured |
| 5% | vector | 34.1815 | 40.3558 | [40.3558](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/5p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_5p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 1.7200s / 1.9267s | 2.3586s / 2.3292s | 2.7900s / 2.5248s | 154,600 | measured |
| 10% | IDs only | 66.2901 | 172.9183 | [172.9183](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_10p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.8910s / 0.4518s | 1.7226s / 0.6338s | 1.8201s / 0.7458s | 1,000 | measured |
| 10% | scalar label | 246.6871 | 271.1625 | [271.1625](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_10p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.2393s / 0.2875s | 0.2825s / 0.3340s | 0.3402s / 0.3846s | 1,800 | measured |
| 10% | vector | 32.5195 | 34.5492 | [34.5492](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/10p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_10p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 1.8147s / 2.2506s | 2.2467s / 2.5049s | 2.6215s / 2.6865s | 154,600 | measured |
| 20% | IDs only | 65.1676 | 146.1041 | [146.1041](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_20p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.9053s / 0.5354s | 1.7845s / 0.7446s | 1.9289s / 0.9023s | 1,000 | measured |
| 20% | scalar label | 203.2728 | 233.9098 | [233.9098](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_20p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.2908s / 0.3316s | 0.3517s / 0.3893s | 0.3852s / 0.5323s | 1,800 | measured |
| 20% | vector | 29.2532 | 31.7358 | [31.7358](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/20p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_20p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 2.0189s / 2.4500s | 2.3934s / 2.7125s | 2.6524s / 2.7942s | 154,600 | measured |
| 50% | IDs only | 63.1829 | 121.7992 | [121.7992](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_50p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.9326s / 0.6422s | 1.9162s / 0.8538s | 2.2440s / 1.0060s | 1,000 | measured |
| 50% | scalar label | 164.6652 | 196.3058 | [196.3058](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_tiered_1cu_scalar_label_filter_50p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.3596s / 0.3975s | 0.4449s / 0.4714s | 0.5335s / 0.5781s | 1,800 | measured |
| 50% | vector | 26.9741 | 28.5275 | [28.5275](raw_results/zilliz_cloud_tiered_1cu/scalar_label_filter/50p/vector/concurrent_qps/result_20260515_zilliz_cloud_tiered_1cu_scalar_label_filter_50p_vector_topk50_c60_c80_30s_20260515_zillizcloud.json) | 2.1880s / 2.7392s | 2.5486s / 3.0166s | 2.8549s / 3.0328s | 154,600 | measured |

## Zilliz Cloud Capacity 2CU

| Item | Value |
|---|---|
| Product key | `zilliz_cloud_capacity_2cu` |
| Notes | Zilliz Cloud capacity cluster configured as 2 CU for multitenant search. |

### Unfiltered Search

| Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---|---:|---:|---:|---|---|---|---:|---|
| IDs only | 883.9646 | 889.2759 | [889.2759](raw_results/zilliz_cloud_capacity_2cu/unfiltered/na/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_unfiltered_na_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0667s / 0.0874s | 0.0909s / 0.1107s | 0.0956s / 0.1144s | 1,000 | measured |
| scalar label | 864.6916 | 869.3213 | [869.3213](raw_results/zilliz_cloud_capacity_2cu/unfiltered/na/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_unfiltered_na_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0682s / 0.0894s | 0.0927s / 0.1116s | 0.0970s / 0.1159s | 1,800 | measured |
| vector | 368.4559 | 371.0902 | [371.0902](raw_results/zilliz_cloud_capacity_2cu/unfiltered/na/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_unfiltered_na_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1598s / 0.2096s | 0.1716s / 0.2214s | 0.1892s / 0.2299s | 154,600 | measured |

### Integer Filtered Search

| Filter rate | Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 99.9% | IDs only | 2097.8559 | 2109.5110 | [2109.5110](raw_results/zilliz_cloud_capacity_2cu/int_filter/99_9p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_0_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0281s / 0.0368s | 0.0636s / 0.0722s | 0.0729s / 0.0796s | 1,000 | measured |
| 99.9% | scalar label | 2052.2448 | 2068.2014 | [2068.2014](raw_results/zilliz_cloud_capacity_2cu/int_filter/99_9p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_0_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0287s / 0.0375s | 0.0648s / 0.0733s | 0.0735s / 0.0808s | 1,800 | measured |
| 99.9% | vector | 1306.5933 | 1302.4189 | [1306.5933](raw_results/zilliz_cloud_capacity_2cu/int_filter/99_9p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_0_1p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0450s / 0.0596s | 0.0760s / 0.0873s | 0.0887s / 0.0998s | 154,600 | measured |
| 99% | IDs only | 1797.1984 | 1811.6332 | [1811.6332](raw_results/zilliz_cloud_capacity_2cu/int_filter/99p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0328s / 0.0429s | 0.0667s / 0.0777s | 0.0766s / 0.0844s | 1,000 | measured |
| 99% | scalar label | 1754.8887 | 1762.2840 | [1762.2840](raw_results/zilliz_cloud_capacity_2cu/int_filter/99p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0335s / 0.0441s | 0.0667s / 0.0752s | 0.0772s / 0.0843s | 1,800 | measured |
| 99% | vector | 963.3106 | 967.7286 | [967.7286](raw_results/zilliz_cloud_capacity_2cu/int_filter/99p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_1p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0612s / 0.0802s | 0.0894s / 0.1073s | 0.0988s / 0.1144s | 154,600 | measured |
| 90% | IDs only | 1402.5685 | 1409.8924 | [1409.8924](raw_results/zilliz_cloud_capacity_2cu/int_filter/90p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_10p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0420s / 0.0551s | 0.0775s / 0.0891s | 0.0833s / 0.0939s | 1,000 | measured |
| 90% | scalar label | 1358.0840 | 1374.2625 | [1374.2625](raw_results/zilliz_cloud_capacity_2cu/int_filter/90p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_10p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0435s / 0.0567s | 0.0800s / 0.0903s | 0.0840s / 0.0942s | 1,800 | measured |
| 90% | vector | 812.8939 | 828.9358 | [828.9358](raw_results/zilliz_cloud_capacity_2cu/int_filter/90p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_10p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0724s / 0.0937s | 0.1018s / 0.1171s | 0.1086s / 0.1273s | 154,600 | measured |
| 50% | IDs only | 902.9304 | 908.8571 | [908.8571](raw_results/zilliz_cloud_capacity_2cu/int_filter/50p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_50p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0653s / 0.0855s | 0.0911s / 0.1092s | 0.0968s / 0.1160s | 1,000 | measured |
| 50% | scalar label | 886.0709 | 892.0028 | [892.0028](raw_results/zilliz_cloud_capacity_2cu/int_filter/50p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_50p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0666s / 0.0871s | 0.0928s / 0.1096s | 0.0961s / 0.1134s | 1,800 | measured |
| 50% | vector | 490.8798 | 501.5928 | [501.5928](raw_results/zilliz_cloud_capacity_2cu/int_filter/50p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_int_filter_50p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1201s / 0.1549s | 0.1415s / 0.1760s | 0.1501s / 0.1872s | 154,600 | measured |

### Scalar Label Filtered Search

| Filter rate | Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 0.1% | IDs only | 944.0011 | 956.1840 | [956.1840](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0625s / 0.0814s | 0.0845s / 0.1013s | 0.0912s / 0.1114s | 1,000 | measured |
| 0.1% | scalar label | 933.4305 | 938.2192 | [938.2192](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0633s / 0.0828s | 0.0849s / 0.1023s | 0.0913s / 0.1102s | 1,800 | measured |
| 0.1% | vector | 724.3978 | 726.7952 | [726.7952](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_1p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0813s / 0.1068s | 0.1037s / 0.1325s | 0.1113s / 0.1505s | 154,600 | measured |
| 0.2% | IDs only | 926.3096 | 932.9284 | [932.9284](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_2p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0637s / 0.0834s | 0.0854s / 0.1036s | 0.0908s / 0.1111s | 1,000 | measured |
| 0.2% | scalar label | 911.0764 | 918.4657 | [918.4657](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_2p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0646s / 0.0848s | 0.0864s / 0.1039s | 0.0933s / 0.1149s | 1,800 | measured |
| 0.2% | vector | 672.9848 | 680.4028 | [680.4028](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_2p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0874s / 0.1143s | 0.1090s / 0.1447s | 0.1182s / 0.1579s | 154,600 | measured |
| 0.5% | IDs only | 886.6728 | 897.4345 | [897.4345](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_5p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0666s / 0.0866s | 0.0883s / 0.1073s | 0.0927s / 0.1172s | 1,000 | measured |
| 0.5% | scalar label | 866.4546 | 871.5848 | [871.5848](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_5p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0681s / 0.0892s | 0.0897s / 0.1085s | 0.0951s / 0.1198s | 1,800 | measured |
| 0.5% | vector | 583.7694 | 602.1184 | [602.1184](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_0_5p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1010s / 0.1290s | 0.1282s / 0.1617s | 0.1501s / 0.1716s | 154,600 | measured |
| 1% | IDs only | 856.8208 | 873.4970 | [873.4970](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_1p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0688s / 0.0891s | 0.0900s / 0.1095s | 0.0939s / 0.1214s | 1,000 | measured |
| 1% | scalar label | 840.4544 | 854.6074 | [854.6074](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_1p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0701s / 0.0909s | 0.0909s / 0.1108s | 0.1018s / 0.1240s | 1,800 | measured |
| 1% | vector | 553.7885 | 588.2456 | [588.2456](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/1p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_1p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1063s / 0.1322s | 0.1417s / 0.1643s | 0.1673s / 0.1745s | 154,600 | measured |
| 2% | IDs only | 845.8386 | 858.7817 | [858.7817](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_2p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0696s / 0.0905s | 0.0913s / 0.1111s | 0.0986s / 0.1224s | 1,000 | measured |
| 2% | scalar label | 830.2696 | 840.5465 | [840.5465](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_2p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0711s / 0.0926s | 0.0916s / 0.1116s | 0.0960s / 0.1195s | 1,800 | measured |
| 2% | vector | 521.5204 | 570.0569 | [570.0569](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/2p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_2p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1129s / 0.1365s | 0.1496s / 0.1691s | 0.1676s / 0.1803s | 154,600 | measured |
| 5% | IDs only | 820.1050 | 831.0361 | [831.0361](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_5p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0718s / 0.0936s | 0.0925s / 0.1123s | 0.1018s / 0.1238s | 1,000 | measured |
| 5% | scalar label | 807.0955 | 811.2514 | [811.2514](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_5p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0729s / 0.0958s | 0.0926s / 0.1136s | 0.0978s / 0.1271s | 1,800 | measured |
| 5% | vector | 466.6632 | 516.8106 | [516.8106](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/5p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_5p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1263s / 0.1505s | 0.1640s / 0.1732s | 0.1756s / 0.1820s | 154,600 | measured |
| 10% | IDs only | 751.3605 | 763.7163 | [763.7163](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_10p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0784s / 0.1017s | 0.0967s / 0.1235s | 0.1004s / 0.1382s | 1,000 | measured |
| 10% | scalar label | 739.1259 | 742.0147 | [742.0147](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_10p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0797s / 0.1047s | 0.0990s / 0.1267s | 0.1042s / 0.1445s | 1,800 | measured |
| 10% | vector | 405.6364 | 434.6915 | [434.6915](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/10p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_10p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1455s / 0.1791s | 0.1778s / 0.1953s | 0.1864s / 0.2036s | 154,600 | measured |
| 20% | IDs only | 707.8775 | 719.5802 | [719.5802](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_20p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0834s / 0.1081s | 0.1005s / 0.1326s | 0.1098s / 0.1464s | 1,000 | measured |
| 20% | scalar label | 697.4468 | 695.7421 | [697.4468](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_20p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.0846s / 0.1112s | 0.1025s / 0.1358s | 0.1071s / 0.1483s | 1,800 | measured |
| 20% | vector | 362.7374 | 376.0030 | [376.0030](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/20p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_20p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1627s / 0.2069s | 0.1835s / 0.2199s | 0.1940s / 0.2268s | 154,600 | measured |
| 50% | IDs only | 579.8026 | 582.0859 | [582.0859](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_50p_ids_only_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1017s / 0.1339s | 0.1203s / 0.1615s | 0.1338s / 0.1742s | 1,000 | measured |
| 50% | scalar label | 569.5811 | 576.5301 | [576.5301](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_50p_scalar_label_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1035s / 0.1350s | 0.1218s / 0.1624s | 0.1352s / 0.1704s | 1,800 | measured |
| 50% | vector | 312.3402 | 317.4757 | [317.4757](raw_results/zilliz_cloud_capacity_2cu/scalar_label_filter/50p/vector/concurrent_qps/result_20260513_zilliz_cloud_capacity_2cu_scalar_label_filter_50p_vector_topk50_c60_c80_30s_20260513_zillizcloud.json) | 0.1889s / 0.2454s | 0.2023s / 0.2587s | 0.2081s / 0.2740s | 154,600 | measured |

## Turbopuffer

| Item | Value |
|---|---|
| Product key | `turbopuffer` |
| Notes | Turbopuffer unpinned multitenant namespaces. |

### Unfiltered Search

| Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---|---:|---:|---:|---|---|---|---:|---|
| IDs only | 2967.4479 | 3854.5229 | [3854.5229](raw_results/turbopuffer/unfiltered/na/ids_only/concurrent_qps/result_20260513_turbopuffer_unfiltered_na_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0199s / 0.0202s | 0.0333s / 0.0343s | 0.0736s / 0.0580s | 1,000 | measured |
| scalar label | 3041.1076 | 3910.8024 | [3910.8024](raw_results/turbopuffer/unfiltered/na/scalar_label/concurrent_qps/result_20260513_turbopuffer_unfiltered_na_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0194s / 0.0199s | 0.0323s / 0.0342s | 0.0522s / 0.0558s | 1,800 | measured |
| vector | 1756.2500 | 1775.3678 | [1775.3678](raw_results/turbopuffer/unfiltered/na/vector/concurrent_qps/result_20260513_turbopuffer_unfiltered_na_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0334s / 0.0436s | 0.0515s / 0.0672s | 0.0794s / 0.0864s | 154,600 | measured |

### Integer Filtered Search

| Filter rate | Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 99.9% | IDs only | 3541.9115 | 4374.4112 | [4374.4112](raw_results/turbopuffer/int_filter/99_9p/ids_only/concurrent_qps/result_20260513_turbopuffer_int_filter_0_1p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0166s / 0.0178s | 0.0265s / 0.0293s | 0.0455s / 0.0501s | 1,000 | measured |
| 99.9% | scalar label | 3462.6385 | 4461.7273 | [4461.7273](raw_results/turbopuffer/int_filter/99_9p/scalar_label/concurrent_qps/result_20260513_turbopuffer_int_filter_0_1p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0170s / 0.0174s | 0.0279s / 0.0291s | 0.0477s / 0.0484s | 1,800 | measured |
| 99.9% | vector | 3063.3113 | 3835.0336 | [3835.0336](raw_results/turbopuffer/int_filter/99_9p/vector/concurrent_qps/result_20260513_turbopuffer_int_filter_0_1p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0192s / 0.0202s | 0.0307s / 0.0306s | 0.0535s / 0.0522s | 154,600 | measured |
| 99% | IDs only | 3498.0448 | 4512.9687 | [4512.9687](raw_results/turbopuffer/int_filter/99p/ids_only/concurrent_qps/result_20260513_turbopuffer_int_filter_1p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0168s / 0.0173s | 0.0264s / 0.0267s | 0.0466s / 0.0461s | 1,000 | measured |
| 99% | scalar label | 3416.8382 | 4285.4931 | [4285.4931](raw_results/turbopuffer/int_filter/99p/scalar_label/concurrent_qps/result_20260513_turbopuffer_int_filter_1p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0172s / 0.0182s | 0.0281s / 0.0295s | 0.0477s / 0.0499s | 1,800 | measured |
| 99% | vector | 1759.4409 | 1754.4688 | [1759.4409](raw_results/turbopuffer/int_filter/99p/vector/concurrent_qps/result_20260513_turbopuffer_int_filter_1p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0333s / 0.0439s | 0.0501s / 0.0674s | 0.0702s / 0.0873s | 154,600 | measured |
| 90% | IDs only | 3332.2455 | 4150.4113 | [4150.4113](raw_results/turbopuffer/int_filter/90p/ids_only/concurrent_qps/result_20260513_turbopuffer_int_filter_10p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0177s / 0.0188s | 0.0289s / 0.0319s | 0.0472s / 0.0578s | 1,000 | measured |
| 90% | scalar label | 3319.6874 | 4302.2748 | [4302.2748](raw_results/turbopuffer/int_filter/90p/scalar_label/concurrent_qps/result_20260513_turbopuffer_int_filter_10p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0177s / 0.0181s | 0.0290s / 0.0295s | 0.0516s / 0.0505s | 1,800 | measured |
| 90% | vector | 1759.9299 | 1767.4318 | [1767.4318](raw_results/turbopuffer/int_filter/90p/vector/concurrent_qps/result_20260513_turbopuffer_int_filter_10p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0333s / 0.0438s | 0.0500s / 0.0682s | 0.0692s / 0.0900s | 154,600 | measured |
| 50% | IDs only | 3165.8892 | 4065.1862 | [4065.1862](raw_results/turbopuffer/int_filter/50p/ids_only/concurrent_qps/result_20260513_turbopuffer_int_filter_50p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0186s / 0.0192s | 0.0310s / 0.0334s | 0.0548s / 0.0535s | 1,000 | measured |
| 50% | scalar label | 3173.9046 | 4143.3079 | [4143.3079](raw_results/turbopuffer/int_filter/50p/scalar_label/concurrent_qps/result_20260513_turbopuffer_int_filter_50p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0185s / 0.0188s | 0.0303s / 0.0317s | 0.0524s / 0.0504s | 1,800 | measured |
| 50% | vector | 1760.4148 | 1759.7066 | [1760.4148](raw_results/turbopuffer/int_filter/50p/vector/concurrent_qps/result_20260513_turbopuffer_int_filter_50p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0333s / 0.0438s | 0.0493s / 0.0678s | 0.0655s / 0.0852s | 154,600 | measured |

### Scalar Label Filtered Search

| Filter rate | Payload | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 0.1% | IDs only | 3462.3573 | 4474.6276 | [4474.6276](raw_results/turbopuffer/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_1p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0170s / 0.0174s | 0.0283s / 0.0283s | 0.0466s / 0.0480s | 1,000 | measured |
| 0.1% | scalar label | 3401.2418 | 4461.1256 | [4461.1256](raw_results/turbopuffer/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_1p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0173s / 0.0174s | 0.0298s / 0.0282s | 0.0487s / 0.0477s | 1,800 | measured |
| 0.1% | vector | 3229.7497 | 3863.0166 | [3863.0166](raw_results/turbopuffer/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_1p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0182s / 0.0201s | 0.0264s / 0.0298s | 0.0458s / 0.0496s | 154,600 | measured |
| 0.2% | IDs only | 3645.1467 | 4421.5355 | [4421.5355](raw_results/turbopuffer/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_2p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0162s / 0.0176s | 0.0242s / 0.0281s | 0.0445s / 0.0517s | 1,000 | measured |
| 0.2% | scalar label | 3431.1940 | 4349.2793 | [4349.2793](raw_results/turbopuffer/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_2p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0172s / 0.0179s | 0.0269s / 0.0299s | 0.0501s / 0.0524s | 1,800 | measured |
| 0.2% | vector | 2875.5294 | 3063.5334 | [3063.5334](raw_results/turbopuffer/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_2p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0204s / 0.0253s | 0.0307s / 0.0370s | 0.0512s / 0.0574s | 154,600 | measured |
| 0.5% | IDs only | 3454.8483 | 4534.6206 | [4534.6206](raw_results/turbopuffer/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_5p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0170s / 0.0172s | 0.0277s / 0.0256s | 0.0476s / 0.0479s | 1,000 | measured |
| 0.5% | scalar label | 3550.5698 | 4559.5927 | [4559.5927](raw_results/turbopuffer/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_5p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0166s / 0.0171s | 0.0250s / 0.0259s | 0.0435s / 0.0439s | 1,800 | measured |
| 0.5% | vector | 1834.7427 | 1840.0142 | [1840.0142](raw_results/turbopuffer/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_0_5p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0320s / 0.0419s | 0.0484s / 0.0643s | 0.0665s / 0.0813s | 154,600 | measured |
| 1% | IDs only | 3453.7997 | 4566.9943 | [4566.9943](raw_results/turbopuffer/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_1p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0171s / 0.0170s | 0.0281s / 0.0255s | 0.0485s / 0.0447s | 1,000 | measured |
| 1% | scalar label | 3509.2407 | 4440.5302 | [4440.5302](raw_results/turbopuffer/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_1p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0168s / 0.0175s | 0.0254s / 0.0276s | 0.0456s / 0.0480s | 1,800 | measured |
| 1% | vector | 1767.2513 | 1762.5671 | [1767.2513](raw_results/turbopuffer/scalar_label_filter/1p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_1p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0332s / 0.0437s | 0.0491s / 0.0674s | 0.0688s / 0.0859s | 154,600 | measured |
| 2% | IDs only | 3463.7862 | 4514.8948 | [4514.8948](raw_results/turbopuffer/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_2p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0170s / 0.0172s | 0.0272s / 0.0270s | 0.0472s / 0.0457s | 1,000 | measured |
| 2% | scalar label | 3501.1969 | 4433.2559 | [4433.2559](raw_results/turbopuffer/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_2p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0168s / 0.0176s | 0.0264s / 0.0277s | 0.0455s / 0.0474s | 1,800 | measured |
| 2% | vector | 1762.3361 | 1759.5860 | [1762.3361](raw_results/turbopuffer/scalar_label_filter/2p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_2p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0333s / 0.0439s | 0.0497s / 0.0674s | 0.0677s / 0.0843s | 154,600 | measured |
| 5% | IDs only | 3505.5387 | 4569.5414 | [4569.5414](raw_results/turbopuffer/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_5p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0168s / 0.0170s | 0.0261s / 0.0262s | 0.0440s / 0.0427s | 1,000 | measured |
| 5% | scalar label | 3441.1856 | 4444.1858 | [4444.1858](raw_results/turbopuffer/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_5p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0171s / 0.0175s | 0.0286s / 0.0265s | 0.0485s / 0.0468s | 1,800 | measured |
| 5% | vector | 1765.0279 | 1765.9335 | [1765.9335](raw_results/turbopuffer/scalar_label_filter/5p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_5p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0332s / 0.0438s | 0.0490s / 0.0687s | 0.0663s / 0.0899s | 154,600 | measured |
| 10% | IDs only | 3422.0162 | 4436.1178 | [4436.1178](raw_results/turbopuffer/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_10p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0172s / 0.0175s | 0.0287s / 0.0274s | 0.0466s / 0.0452s | 1,000 | measured |
| 10% | scalar label | 3390.0875 | 4352.6850 | [4352.6850](raw_results/turbopuffer/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_10p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0174s / 0.0179s | 0.0284s / 0.0285s | 0.0475s / 0.0490s | 1,800 | measured |
| 10% | vector | 1757.2460 | 1759.4816 | [1759.4816](raw_results/turbopuffer/scalar_label_filter/10p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_10p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0334s / 0.0439s | 0.0505s / 0.0684s | 0.0690s / 0.0903s | 154,600 | measured |
| 20% | IDs only | 3056.6678 | 4045.5609 | [4045.5609](raw_results/turbopuffer/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_20p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0192s / 0.0193s | 0.0362s / 0.0347s | 0.0581s / 0.0550s | 1,000 | measured |
| 20% | scalar label | 3176.0501 | 4063.1335 | [4063.1335](raw_results/turbopuffer/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_20p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0186s / 0.0192s | 0.0323s / 0.0341s | 0.0536s / 0.0543s | 1,800 | measured |
| 20% | vector | 1750.8508 | 1762.2420 | [1762.2420](raw_results/turbopuffer/scalar_label_filter/20p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_20p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0335s / 0.0438s | 0.0542s / 0.0682s | 0.0785s / 0.0877s | 154,600 | measured |
| 50% | IDs only | 1491.2085 | 4147.7988 | [4147.7988](raw_results/turbopuffer/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_50p_ids_only_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0202s / 0.0189s | 0.0371s / 0.0306s | 0.0600s / 0.0551s | 1,000 | measured |
| 50% | scalar label | 3182.9416 | 4219.6587 | [4219.6587](raw_results/turbopuffer/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_50p_scalar_label_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0185s / 0.0184s | 0.0303s / 0.0283s | 0.0520s / 0.0462s | 1,800 | measured |
| 50% | vector | 1755.3520 | 1759.7100 | [1759.7100](raw_results/turbopuffer/scalar_label_filter/50p/vector/concurrent_qps/result_20260513_turbopuffer_scalar_label_filter_50p_vector_topk50_c60_c80_30s_20260513_turbopuffer.json) | 0.0334s / 0.0438s | 0.0502s / 0.0677s | 0.0689s / 0.0866s | 154,600 | measured |

## Pinecone Serverless

| Item | Value |
|---|---|
| Product key | `pinecone_serverless` |
| Notes | Pinecone Serverless multitenant namespace results were collected in a detached c4-only run. They are included for traceability but are not directly comparable with the c60/c80 rows above. |

### Unfiltered Search

| Payload | QPS @4 | Max QPS | Avg latency @4 | P95 @4 | P99 @4 | Payload bytes/query | Status |
|---|---:|---:|---|---|---|---:|---|
| IDs only | 568.9403 | [568.9403](raw_results/pinecone_serverless/unfiltered/na/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_unfiltered_na_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0070s | 0.0117s | 0.0364s | 1,000 | measured c4 |
| vector | 541.9661 | [541.9661](raw_results/pinecone_serverless/unfiltered/na/vector/concurrent_qps/result_20260514_pinecone_multitenant_unfiltered_na_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0073s | 0.0111s | 0.0403s | 154,600 | measured c4 |

### Integer Filtered Search

| Filter rate | Payload | QPS @4 | Max QPS | Avg latency @4 | P95 @4 | P99 @4 | Payload bytes/query | Status |
|---:|---|---:|---:|---|---|---|---:|---|
| 99.9% | IDs only | 524.5527 | [524.5527](raw_results/pinecone_serverless/int_filter/99_9p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_0_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0076s | 0.0133s | 0.0390s | 1,000 | measured c4 |
| 99.9% | vector | 526.1875 | [526.1875](raw_results/pinecone_serverless/int_filter/99_9p/vector/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_0_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0075s | 0.0109s | 0.0374s | 154,600 | measured c4 |
| 99% | IDs only | 549.3702 | [549.3702](raw_results/pinecone_serverless/int_filter/99p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0072s | 0.0124s | 0.0372s | 1,000 | measured c4 |
| 99% | vector | 571.7549 | [571.7549](raw_results/pinecone_serverless/int_filter/99p/vector/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0069s | 0.0101s | 0.0362s | 154,600 | measured c4 |
| 90% | IDs only | 529.5485 | [529.5485](raw_results/pinecone_serverless/int_filter/90p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_10p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0075s | 0.0133s | 0.0391s | 1,000 | measured c4 |
| 90% | vector | 502.3192 | [502.3192](raw_results/pinecone_serverless/int_filter/90p/vector/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_10p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0079s | 0.0147s | 0.0450s | 154,600 | measured c4 |
| 50% | IDs only | 551.2294 | [551.2294](raw_results/pinecone_serverless/int_filter/50p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_50p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0072s | 0.0110s | 0.0325s | 1,000 | measured c4 |
| 50% | vector | 506.8155 | [506.8155](raw_results/pinecone_serverless/int_filter/50p/vector/concurrent_qps/result_20260514_pinecone_multitenant_int_filter_50p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0078s | 0.0149s | 0.0409s | 154,600 | measured c4 |

### Scalar Label Filtered Search

| Filter rate | Payload | QPS @4 | Max QPS | Avg latency @4 | P95 @4 | P99 @4 | Payload bytes/query | Status |
|---:|---|---:|---:|---|---|---|---:|---|
| 0.1% | IDs only | 565.6945 | [565.6945](raw_results/pinecone_serverless/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_0_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0070s | 0.0095s | 0.0272s | 1,000 | measured c4 |
| 0.1% | vector | 490.4772 | [490.4772](raw_results/pinecone_serverless/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_0_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0081s | 0.0169s | 0.0425s | 154,600 | measured c4 |
| 0.2% | IDs only | 559.9415 | [559.9415](raw_results/pinecone_serverless/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_0_2p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0071s | 0.0102s | 0.0306s | 1,000 | measured c4 |
| 0.2% | vector | 534.4986 | [534.4986](raw_results/pinecone_serverless/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_0_2p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0074s | 0.0118s | 0.0436s | 154,600 | measured c4 |
| 0.5% | IDs only | 556.1593 | [556.1593](raw_results/pinecone_serverless/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_0_5p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0071s | 0.0104s | 0.0336s | 1,000 | measured c4 |
| 0.5% | vector | 518.5324 | [518.5324](raw_results/pinecone_serverless/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_0_5p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0076s | 0.0122s | 0.0432s | 154,600 | measured c4 |
| 1% | IDs only | 566.5200 | [566.5200](raw_results/pinecone_serverless/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0070s | 0.0095s | 0.0224s | 1,000 | measured c4 |
| 1% | vector | 600.1535 | [600.1535](raw_results/pinecone_serverless/scalar_label_filter/1p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0066s | 0.0093s | 0.0348s | 154,600 | measured c4 |
| 2% | IDs only | 547.7551 | [547.7551](raw_results/pinecone_serverless/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_2p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0072s | 0.0107s | 0.0314s | 1,000 | measured c4 |
| 2% | vector | 538.2404 | [538.2404](raw_results/pinecone_serverless/scalar_label_filter/2p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_2p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0074s | 0.0112s | 0.0439s | 154,600 | measured c4 |
| 5% | IDs only | 565.9264 | [565.9264](raw_results/pinecone_serverless/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_5p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0070s | 0.0102s | 0.0343s | 1,000 | measured c4 |
| 5% | vector | 588.4483 | [588.4483](raw_results/pinecone_serverless/scalar_label_filter/5p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_5p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0067s | 0.0090s | 0.0309s | 154,600 | measured c4 |
| 10% | IDs only | 573.9603 | [573.9603](raw_results/pinecone_serverless/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_10p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0069s | 0.0105s | 0.0336s | 1,000 | measured c4 |
| 10% | vector | 538.9582 | [538.9582](raw_results/pinecone_serverless/scalar_label_filter/10p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_10p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0074s | 0.0122s | 0.0433s | 154,600 | measured c4 |
| 20% | IDs only | 483.9911 | [483.9911](raw_results/pinecone_serverless/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_20p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0082s | 0.0171s | 0.0505s | 1,000 | measured c4 |
| 20% | vector | 517.0640 | [517.0640](raw_results/pinecone_serverless/scalar_label_filter/20p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_20p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0077s | 0.0133s | 0.0444s | 154,600 | measured c4 |
| 50% | IDs only | 494.2505 | [494.2505](raw_results/pinecone_serverless/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_50p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0080s | 0.0134s | 0.0485s | 1,000 | measured c4 |
| 50% | vector | 562.3003 | [562.3003](raw_results/pinecone_serverless/scalar_label_filter/50p/vector/concurrent_qps/result_20260514_pinecone_multitenant_scalar_label_filter_50p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.0071s | 0.0104s | 0.0342s | 154,600 | measured c4 |
