# Cloud Cold Latency

## Scope

LAION 100M, 768 dimensions, L2, topK 100, ids-only response payload, 1,000 serial queries per cold pass and 1,000 serial queries per warm pass.

## Results

| Product | Mode | Collection | First cold query (s) | Cold p99 (s) | Cold p95 (s) | Cold avg (s) | Warm p99 (s) | Warm p95 (s) | Warm avg (s) | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Zilliz Cloud Capacity 12CU | Unfiltered | `LAION100M_capacity` | 0.1151 | 0.0228 | 0.0143 | 0.0115 | 0.0182 | 0.0137 | 0.0109 | Accepted |
| Zilliz Cloud Capacity 12CU | Int filter 0.9 | `LAION100M_capacity` | 0.1951 | 0.0155 | 0.0133 | 0.0114 | 0.0144 | 0.0131 | 0.0112 | Accepted |
| Zilliz Cloud Tiered 4CU | Unfiltered | `LAION100M` | 0.1222 | 0.4867 | 0.2108 | 0.0764 | 0.0655 | 0.0495 | 0.0279 | Rebench needed |
| Zilliz Cloud Tiered 4CU | Int filter 0.9 | `LAION100M` | 2.4380 | 0.1468 | 0.0388 | 0.0262 | 0.0161 | 0.0117 | 0.0105 | Rebench needed |
| Pinecone Serverless | Unfiltered | `vdbbench-laion-100m-768d-l2` | 0.2713 | 1.0925 | 1.0702 | 0.2330 | 1.0894 | 1.0673 | 0.2475 | Accepted |
| Pinecone Serverless | Int filter 0.9 | `vdbbench-laion-100m-768d-l2` | 0.9106 | 1.1424 | 0.6552 | 0.2370 | 1.1493 | 0.6602 | 0.2473 | Accepted |
| Turbopuffer | Unfiltered | `laion100m_bulk` | 2.0476 | 0.6822 | 0.5596 | 0.4145 | 0.7041 | 0.5434 | 0.3933 | Accepted |
| Turbopuffer | Int filter 0.9 | `laion100m_bulk` | 4.4686 | 1.3141 | 0.9609 | 0.6490 | 0.7802 | 0.6870 | 0.5535 | Accepted |
| Turbopuffer Pinned 2 Replicas | Unfiltered | `laion100m_bulk` | 0.0642 | 0.0583 | 0.0553 | 0.0459 | 0.0573 | 0.0538 | 0.0450 | Accepted |
| Turbopuffer Pinned 2 Replicas | Int filter 0.9 | `laion100m_bulk` | 0.0842 | 0.0734 | 0.0701 | 0.0630 | 0.0715 | 0.0694 | 0.0629 | Accepted |

## Notes

- Capacity 12CU looks internally consistent: cold and warm averages are close, with the main cold penalty concentrated on the first query.
- Tiered 4CU is recorded in raw results, but flagged for rebench because its cold tail is much higher than warm and materially higher than capacity 12CU.
- The 2026-05-15 int-filter 0.9 runs replace the earlier zero-filled placeholder artifacts for Zilliz Cloud Capacity 12CU, Zilliz Cloud Tiered 4CU, Pinecone Serverless, and Turbopuffer.
- Pinecone Serverless unfiltered was rerun on 2026-05-14 and replaces the earlier zero-filled placeholder artifact for reporting. The first-query cold/warm ratio is 4.5217x; p99 is 1.0028x, p95 is 1.0027x, and average latency is 0.9414x. Raw VDBBench output is archived at `raw_outputs/pinecone_cloud_cold_latency_laion100m_unfiltered_20260514.log`.
- Turbopuffer shows a large first-query cold penalty, while cold and warm tail/average latency are close after the first query.
- Turbopuffer pinned with 2 replicas removes most of the cold-start penalty: unfiltered cold p99 is 0.0583s versus warm p99 0.0573s, and int-filter 0.9 cold p99 is 0.0734s versus warm p99 0.0715s.
