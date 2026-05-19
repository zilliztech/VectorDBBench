# Single Tenant 100M Search Matrix

## Purpose

This document is the execution template and result tracker for exhaustive
single-tenant LAION 100M search testing across cloud vector database products.
The matrix is designed to separate three effects:

- Whether the search is unfiltered or filtered.
- Which filter type is used: integer range filter or string scalar label filter.
- Which payload is returned: IDs only, scalar label field, or vector field.
- What recall/ndcg each exact product, filter, and payload combination gets
  before measuring concurrent QPS.

All rows should use the same dataset shape unless a product cannot support it:
LAION 100M, 768 dimensions, L2, topK 100, with a serial recall phase followed
by a concurrent QPS phase unless noted.

## Table of Contents

- [Execution Matrix](#execution-matrix)
- [Common Parameters](#common-parameters)
- [VDBBench Invocation](#vdbbench-invocation)
- [Result JSON Interpretation](#result-json-interpretation)
- [Zilliz Cloud Tiered 4CU](#zilliz-cloud-tiered-4cu)
- [Zilliz Cloud Capacity 12CU](#zilliz-cloud-capacity-12cu)
- [Pinecone Serverless](#pinecone-serverless)
- [Turbopuffer Unpinned](#turbopuffer-unpinned)
- [Turbopuffer Pinned](#turbopuffer-pinned)
- [Run Queue](#run-queue)

## Execution Matrix

Each product section should eventually contain this full matrix.

| Search mode | Filter flag | Filter expression | Payload profile | Required phases | Status |
|---|---|---|---|---|---|
| Unfiltered | none | none | IDs only | serial recall, then concurrent QPS | pending |
| Unfiltered | none | none | scalar label | serial recall, then concurrent QPS | pending |
| Unfiltered | none | none | vector | serial recall, then concurrent QPS | pending |
| Integer filtered | `--cloud-filter-rate <rate>` | `id >= int(dataset_size * rate)` | IDs only | serial recall, then concurrent QPS | pending |
| Integer filtered | `--cloud-filter-rate <rate>` | `id >= int(dataset_size * rate)` | scalar label | serial recall, then concurrent QPS | pending |
| Integer filtered | `--cloud-filter-rate <rate>` | `id >= int(dataset_size * rate)` | vector | serial recall, then concurrent QPS | pending |
| Scalar label filtered | `--cloud-label-percentage <rate>` | `label == "label_<rate>"` | IDs only | serial recall, then concurrent QPS | Tiered 4CU 1% throughput measured; recall pending |
| Scalar label filtered | `--cloud-label-percentage <rate>` | `label == "label_<rate>"` | scalar label | serial recall, then concurrent QPS | Tiered 4CU 1% throughput measured; recall pending |
| Scalar label filtered | `--cloud-label-percentage <rate>` | `label == "label_<rate>"` | vector | serial recall, then concurrent QPS | Tiered 4CU 1% throughput measured; recall pending |

Planned filter rates:

| Filter type | Candidate rates |
|---|---|
| Integer filter | `0.999`, `0.99`, `0.9`, `0.5` |
| Scalar label filter | `0.001`, `0.002`, `0.005`, `0.01`, `0.02`, `0.05`, `0.1`, `0.2`, `0.5` |

## Common Parameters

| Item | Value |
|---|---|
| Dataset | LAION 100M |
| Dimensions | 768 |
| Metric | L2 |
| TopK | 100 |
| Search phases | serial recall first, then concurrent QPS |
| Default concurrency list | `60,80` |
| Default duration | 60s per concurrency |
| Serial search | required for every product and every matrix row |
| Result repo | `cloud_leadboard_tests_0509` |
| VDBBench repo | `/home/ubuntu/vdbbenchleadboard2/VectorDBBench` |
| VDBBench branch | `cloud-payload-search-case` |
| Current framework commit | `2183232c0e718e64e282c8b1c51de49309dc1128` plus local cloud scalar-label changes until committed |

Payload estimates currently emitted by VDBBench:

| Payload profile | Meaning | Estimated bytes/query at topK 100 |
|---|---|---:|
| `ids_only` | primary key plus distance | 2,000 |
| `scalar_label` | primary key, distance, and label string | 3,600 |
| `vector` | primary key, distance, and 768D float vector | 309,200 |

## VDBBench Invocation

Set credentials outside the repository. Do not commit credentials or private
service URLs.

```bash
export DATASET_LOCAL_DIR=/mnt/instance/vectordb_bench/dataset
export ZILLIZ_PASSWORD='<password>'
export ZILLIZ_TOKEN='<token>'
```

Every case needs two runs. Run serial search first to capture recall for
the exact product, filter, and payload combination. Then run concurrent search
for QPS and latency.

Base Zilliz serial recall command:

```bash
.venv/bin/python -X faulthandler -m vectordb_bench.cli.vectordbbench zillizautoindex \
  --uri '<zilliz-uri>' \
  --user-name db_admin \
  --case-type CloudPayloadSearchCase \
  --payload-profile '<ids_only|scalar_label|vector>' \
  --collection-name '<collection-name>' \
  --skip-drop-old --skip-load \
  --search-serial --skip-search-concurrent \
  --db-label '<run-label>_serial_recall'
```

Base Zilliz concurrent QPS command:

```bash
.venv/bin/python -X faulthandler -m vectordb_bench.cli.vectordbbench zillizautoindex \
  --uri '<zilliz-uri>' \
  --user-name db_admin \
  --case-type CloudPayloadSearchCase \
  --payload-profile '<ids_only|scalar_label|vector>' \
  --collection-name '<collection-name>' \
  --skip-drop-old --skip-load \
  --skip-search-serial --search-concurrent \
  --num-concurrency 60,80 \
  --concurrency-duration 60 \
  --db-label '<run-label>'
```

Add exactly one filter flag for filtered runs:

```bash
# Integer filter.
--cloud-filter-rate 0.01

# Scalar string label filter.
--cloud-label-percentage 0.01
```

Product-specific commands for Pinecone and Turbopuffer should be filled after
their CloudPayloadSearchCase support and collection layout are validated.

Do not combine serial and concurrent in one run for this matrix. The current
runner initializes both stages together and executes concurrent search before
serial search when both are enabled, while this test plan requires serial recall
to be captured first.

## Result JSON Interpretation

Each run produces a JSON file under:

```text
vectordb_bench/results/<Product>/result_<date>_<run_id>_<product>.json
```

Read these fields:

| JSON path | Meaning |
|---|---|
| `run_id` | Unique run id used in the result filename |
| `results[0].task_config.db_config.db_label` | Human-readable run label |
| `results[0].task_config.case_config.custom_case.payload_profile` | Payload profile requested |
| `results[0].task_config.case_config.custom_case.filter_rate` | Integer filter rate, if present |
| `results[0].task_config.case_config.custom_case.label_percentage` | Scalar label filter rate, if present |
| `results[0].metrics.qps` | Maximum QPS across the tested concurrency list |
| `results[0].metrics.conc_num_list` | Concurrency levels tested |
| `results[0].metrics.conc_qps_list` | QPS at each concurrency level |
| `results[0].metrics.conc_latency_avg_list` | Average latency at each concurrency level |
| `results[0].metrics.conc_latency_p95_list` | P95 latency at each concurrency level |
| `results[0].metrics.conc_latency_p99_list` | P99 latency at each concurrency level |
| `results[0].metrics.payload_estimated_bytes_per_query` | Estimated returned bytes/query |
| `results[0].metrics.recall`, `results[0].metrics.ndcg` | Recall and NDCG from the serial run |
| `results[0].metrics.serial_latency_p99`, `results[0].metrics.serial_latency_p95` | Serial latency from the recall run |

A case is complete only when both artifacts are recorded:

| Artifact | Required contents |
|---|---|
| Serial recall JSON | recall, ndcg, serial p95, serial p99 |
| Concurrent QPS JSON | max QPS, per-concurrency QPS, average/p95/p99 latency |

Raw JSON outputs copied into this repository are stored under
`cloud_payload_search/raw_results/` and indexed by
`cloud_payload_search/raw_results/manifest.jsonl`. Use the manifest as the machine-readable
source for official leaderboard ingestion. In the result tables, the `Recall`
cell links to the serial recall JSON when that artifact is present; the
`Max QPS` cell links to the concurrent throughput JSON.

## Zilliz Cloud Tiered 4CU

| Item | Value |
|---|---|
| Collection | `LAION100M` |
| Scalar label field | `label` |
| Scalar label index | `labels_idx` |
| Logical row count | 100,000,000 |

### Unfiltered Search

| Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| IDs only | [0.951](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/serial_recall/result_20260511_6342b75c2e3a42f08701f5102d4d627d_zillizcloud.json) | 44.8981 | 49.1625 | [49.1625](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/concurrent_qps/result_20260511_f3b2fc7a2a864a91a570bd623c1b57a1_zillizcloud.json) | 1.3286s / 1.6187s | 1.4682s / 1.8900s | 5.5715s / 2.2987s | measured |
| scalar label | [0.951](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/scalar_label/serial_recall/result_20260511_93415d2ccee247e390e6d75bc720937e_zillizcloud.json) | 49.0737 | 49.8156 | [49.8156](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/scalar_label/concurrent_qps/result_20260511_37330a1870d54c81af133464d39ae6a1_zillizcloud.json) | 1.2135s / 1.5885s | 1.4760s / 1.9036s | 1.6034s / 2.0174s | measured |
| vector | [0.951](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/vector/serial_recall/result_20260511_4d7523cc67104f3bb8bf87ecf8252ab3_zillizcloud.json) | 32.8253 | 44.0385 | [44.0385](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/vector/concurrent_qps/result_20260509_6b3cbbfcc62d4752b1afbdc2f0874ee3_zillizcloud.json) | 1.8196s / 1.7981s | 2.2119s / 2.0263s | 11.5227s / 2.7943s | measured |

### Integer Filtered Search

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 99.9% | IDs only | [0.9423](raw_results/zilliz_cloud_tiered_4cu/int_filter/0_1p/ids_only/serial_recall/result_20260511_2c060c4fbfe24d4594bd7fca2939c652_zillizcloud.json) | 783.3761 | 948.9499 | [948.9499](raw_results/zilliz_cloud_tiered_4cu/int_filter/0_1p/ids_only/concurrent_qps/result_20260511_07e226c8a9fc4f5b902c690863036d4f_zillizcloud.json) | 0.0760s / 0.0835s | 0.1022s / 0.1075s | 0.1141s / 0.1562s | measured |
| 99.9% | scalar label | [0.9423](raw_results/zilliz_cloud_tiered_4cu/int_filter/0_1p/scalar_label/serial_recall/result_20260511_f8a47e265ab84ff4b4ee4ab2842dc02b_zillizcloud.json) | 764.5844 | 940.5247 | [940.5247](raw_results/zilliz_cloud_tiered_4cu/int_filter/0_1p/scalar_label/concurrent_qps/result_20260511_c1046be4c1364e29bb617a3688d74517_zillizcloud.json) | 0.0779s / 0.0842s | 0.1029s / 0.1077s | 0.1122s / 0.1375s | measured |
| 99.9% | vector | [0.9423](raw_results/zilliz_cloud_tiered_4cu/int_filter/0_1p/vector/serial_recall/result_20260511_b2e7e9ea3c5b41e58a0617a939df1eb1_zillizcloud.json) | 729.0385 | 955.6522 | [955.6522](raw_results/zilliz_cloud_tiered_4cu/int_filter/0_1p/vector/concurrent_qps/result_20260511_afac95e59d814505ad688cb8e47281e5_zillizcloud.json) | 0.0817s / 0.0828s | 0.1079s / 0.1141s | 0.1198s / 0.1604s | measured |
| 99% | IDs only | [0.9557](raw_results/zilliz_cloud_tiered_4cu/int_filter/1p/ids_only/serial_recall/result_20260511_dc5fc933596e4101b9b17716e859f29c_zillizcloud.json) | 587.7227 | 694.9634 | [694.9634](raw_results/zilliz_cloud_tiered_4cu/int_filter/1p/ids_only/concurrent_qps/result_20260511_0145ffcff3574549a792518f4d608fa5_zillizcloud.json) | 0.1014s / 0.1140s | 0.1645s / 0.1752s | 0.1752s / 0.1856s | measured |
| 99% | scalar label | [0.9557](raw_results/zilliz_cloud_tiered_4cu/int_filter/1p/scalar_label/serial_recall/result_20260511_0b4db8da5a68403cb9cd5aa575ae13db_zillizcloud.json) | 581.9005 | 682.8657 | [682.8657](raw_results/zilliz_cloud_tiered_4cu/int_filter/1p/scalar_label/concurrent_qps/result_20260511_dfa96a509e2c4690a3c149b4ba283dac_zillizcloud.json) | 0.1024s / 0.1160s | 0.1656s / 0.1775s | 0.1750s / 0.1868s | measured |
| 99% | vector | [0.9557](raw_results/zilliz_cloud_tiered_4cu/int_filter/1p/vector/serial_recall/result_20260511_4829a5d39a924000b44e20384a67632d_zillizcloud.json) | 515.7132 | 599.7719 | [599.7719](raw_results/zilliz_cloud_tiered_4cu/int_filter/1p/vector/concurrent_qps/result_20260511_a5fd1f999e94429dbe11a9258bbdeea1_zillizcloud.json) | 0.1155s / 0.1319s | 0.1737s / 0.1862s | 0.1857s / 0.1966s | measured |
| 90% | IDs only | [0.9588](raw_results/zilliz_cloud_tiered_4cu/int_filter/10p/ids_only/serial_recall/result_20260511_e36b784ae6804868a4ab22f6f06c0b4f_zillizcloud.json) | 238.9813 | 255.3548 | [255.3548](raw_results/zilliz_cloud_tiered_4cu/int_filter/10p/ids_only/concurrent_qps/result_20260511_7887083605414527bde0a4389a4361d8_zillizcloud.json) | 0.2496s / 0.3107s | 0.3014s / 0.3878s | 0.3745s / 0.4119s | measured |
| 90% | scalar label | [0.9588](raw_results/zilliz_cloud_tiered_4cu/int_filter/10p/scalar_label/serial_recall/result_20260511_356abac3e78e4ca185b60106fe5c2501_zillizcloud.json) | 236.9028 | 254.5785 | [254.5785](raw_results/zilliz_cloud_tiered_4cu/int_filter/10p/scalar_label/concurrent_qps/result_20260511_2070932950a64b17ac353f7bcf0c2c63_zillizcloud.json) | 0.2519s / 0.3117s | 0.3067s / 0.3925s | 0.3717s / 0.4281s | measured |
| 90% | vector | [0.9588](raw_results/zilliz_cloud_tiered_4cu/int_filter/10p/vector/serial_recall/result_20260511_2209c78629b04b70ae39aa02a07f9dc2_zillizcloud.json) | 186.6812 | 194.9450 | [194.9450](raw_results/zilliz_cloud_tiered_4cu/int_filter/10p/vector/concurrent_qps/result_20260511_55fa8ad673364616a337fed2b86b32f2_zillizcloud.json) | 0.3194s / 0.4070s | 0.3928s / 0.4928s | 0.4613s / 0.5582s | measured |
| 50% | IDs only | [0.9543](raw_results/zilliz_cloud_tiered_4cu/int_filter/50p/ids_only/serial_recall/result_20260511_03d11c6197184f7294a2eb4d2840fc97_zillizcloud.json) | 64.0463 | 65.4981 | [65.4981](raw_results/zilliz_cloud_tiered_4cu/int_filter/50p/ids_only/concurrent_qps/result_20260511_54e51504d1794ad7bcd821945cc3068d_zillizcloud.json) | 0.9328s / 1.2088s | 1.1137s / 1.5151s | 1.2168s / 1.7058s | measured |
| 50% | scalar label | [0.9543](raw_results/zilliz_cloud_tiered_4cu/int_filter/50p/scalar_label/serial_recall/result_20260511_1983c0014ccb4ad1bcc91a1610e773db_zillizcloud.json) | 63.0552 | 63.2795 | [63.2795](raw_results/zilliz_cloud_tiered_4cu/int_filter/50p/scalar_label/concurrent_qps/result_20260511_1077ef7629e147ff8f085b69ed0f3b07_zillizcloud.json) | 0.9478s / 1.2528s | 1.1941s / 1.5682s | 1.2929s / 2.0663s | measured |
| 50% | vector | [0.9543](raw_results/zilliz_cloud_tiered_4cu/int_filter/50p/vector/serial_recall/result_20260511_fb0b5da85bf842b8acaa4afe22c0de31_zillizcloud.json) | 48.8324 | 53.0284 | [53.0284](raw_results/zilliz_cloud_tiered_4cu/int_filter/50p/vector/concurrent_qps/result_20260511_69c3e6abf6644d64b0743a6747da7d28_zillizcloud.json) | 1.2206s / 1.4965s | 1.5709s / 1.9855s | 1.8313s / 2.2144s | measured |

### Scalar Label Filtered Search

1% scalar-label filter rate means `label == "label_1p"`, approximately 1M matched rows.

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---|
| 0.1% | IDs only | [0.9742](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_1p/ids_only/serial_recall/result_20260511_eab166e81d9044cd927ee4ce8e03d70d_zillizcloud.json) | 59.3971 | 62.0450 | [62.0450](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260511_7d8889a1b92a48a3a751f1f86eb70341_zillizcloud.json) | 1.0022s / 1.2755s | 1.1732s / 1.5050s | 1.1977s / 1.7010s | 2,000 | measured |
| 0.1% | scalar label | [0.9742](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_1p/scalar_label/serial_recall/result_20260511_1299d20babf9439cb01ce617cbcee2c8_zillizcloud.json) | 56.8579 | 59.8387 | [59.8387](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260511_b25da852b6a549938c7d4c32a7b9c2c8_zillizcloud.json) | 1.0480s / 1.3235s | 1.2133s / 1.5898s | 1.2976s / 1.6870s | 3,600 | measured |
| 0.1% | vector | [0.9742](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_1p/vector/serial_recall/result_20260511_142cadd59e614dc587ba2881cae77824_zillizcloud.json) | 46.0338 | 48.6365 | [48.6365](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260511_7f8a9f44e6be40dcb464cdd288a83fae_zillizcloud.json) | 1.2959s / 1.6284s | 1.5145s / 2.1271s | 1.8824s / 2.3751s | 309,200 | measured |
| 0.2% | IDs only | [0.973](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_2p/ids_only/serial_recall/result_20260511_547830ccd8414df883788927c07471d3_zillizcloud.json) | 65.6970 | 68.3875 | [68.3875](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260511_37a44d4639634c878a6c66e2761a3158_zillizcloud.json) | 0.9097s / 1.1609s | 1.0834s / 1.3947s | 1.1229s / 1.5671s | 2,000 | measured |
| 0.2% | scalar label | [0.973](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_2p/scalar_label/serial_recall/result_20260511_9b25684fb7e14a90b18dcf6594bebc40_zillizcloud.json) | 62.3662 | 65.4546 | [65.4546](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260511_dacadba316e44532b6079206577deee9_zillizcloud.json) | 0.9568s / 1.2139s | 1.1139s / 1.4694s | 1.2918s / 1.5123s | 3,600 | measured |
| 0.2% | vector | [0.973](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_2p/vector/serial_recall/result_20260511_3f44e5dc892e43de8f95389c6ba081de_zillizcloud.json) | 49.6225 | 51.7891 | [51.7891](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260511_86f921f88a8f4b9294cdd24a45c85fc3_zillizcloud.json) | 1.2026s / 1.5329s | 1.4273s / 1.8922s | 1.7129s / 2.0972s | 309,200 | measured |
| 0.5% | IDs only | [0.9708](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_5p/ids_only/serial_recall/result_20260511_4be68683c29c4efab313a7841d7efcae_zillizcloud.json) | 74.0725 | 78.6973 | [78.6973](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260511_9f9f835ed31647f581aea649a32f99cc_zillizcloud.json) | 0.8043s / 1.0116s | 0.9615s / 1.1935s | 1.0306s / 1.3992s | 2,000 | measured |
| 0.5% | scalar label | [0.9708](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_5p/scalar_label/serial_recall/result_20260511_690dec2bf18c4fb9b14c73a55165a62d_zillizcloud.json) | 70.3328 | 74.2955 | [74.2955](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260511_7af190b061f94f459479ca7a42f22222_zillizcloud.json) | 0.8491s / 1.0686s | 1.0178s / 1.2857s | 1.1645s / 1.4285s | 3,600 | measured |
| 0.5% | vector | [0.9708](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_5p/vector/serial_recall/result_20260511_10172556b3a54a2bbd22c3279985c373_zillizcloud.json) | 55.5869 | 57.1731 | [57.1731](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260511_afa468155c99439bacba970bfc508fef_zillizcloud.json) | 1.0731s / 1.3878s | 1.3855s / 1.7128s | 1.5935s / 1.8702s | 309,200 | measured |
| 1% | IDs only | [0.9681](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/1p/ids_only/serial_recall/result_20260511_05852f6bdb884aea8f60a9c23bf5741f_zillizcloud.json) | 81.3103 | 89.6180 | [89.6180](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260511_256d6ebeeaae45a8b269a97c3175b254_zillizcloud.json) | 0.7346s / 0.8844s | 0.8674s / 0.9983s | 2.2083s / 1.1814s | 2,000 | measured |
| 1% | scalar label | [0.9681](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/1p/scalar_label/serial_recall/result_20260511_2dabfa94c7254d68add15f165057b7b2_zillizcloud.json) | 79.6971 | 84.9877 | [84.9877](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260511_9bfaa3afe288417eb9a550605e2affec_zillizcloud.json) | 0.7479s / 0.9323s | 0.8925s / 1.1024s | 1.0073s / 1.2081s | 3,600 | measured |
| 1% | vector | [0.9681](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/1p/vector/serial_recall/result_20260511_c17971d95d8543a69178ed620b6da281_zillizcloud.json) | 57.3345 | 62.8719 | [62.8719](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/1p/vector/concurrent_qps/result_20260511_129cbbc9f61f464193a990d90f239fbb_zillizcloud.json) | 1.0400s / 1.2610s | 1.4088s / 1.5810s | 1.7919s / 1.7895s | 309,200 | measured |
| 2% | IDs only | [0.9654](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/2p/ids_only/serial_recall/result_20260511_545f6eac916d4b0ea5cee313dce448bf_zillizcloud.json) | 93.0085 | 98.4835 | [98.4835](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260511_e3a958226588433384af1b4f34847f39_zillizcloud.json) | 0.6426s / 0.8056s | 0.7774s / 0.9207s | 0.9039s / 1.0018s | 2,000 | measured |
| 2% | scalar label | [0.9654](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/2p/scalar_label/serial_recall/result_20260511_1faa4de6b15b4287b48d8c39d31bf78f_zillizcloud.json) | 87.8265 | 93.0772 | [93.0772](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260511_6b0d386bcc274c2b93753c7053307101_zillizcloud.json) | 0.6793s / 0.8535s | 0.8019s / 1.0197s | 0.9043s / 1.1035s | 3,600 | measured |
| 2% | vector | [0.9654](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/2p/vector/serial_recall/result_20260511_b692e11de6164812a09fafb8edc4fbf8_zillizcloud.json) | 62.4006 | 68.0889 | [68.0889](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/2p/vector/concurrent_qps/result_20260511_1e8e553e0c804b1ca15360bfb2646313_zillizcloud.json) | 0.9559s / 1.1634s | 1.2744s / 1.5180s | 1.4644s / 1.7076s | 309,200 | measured |
| 5% | IDs only | [0.9637](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/5p/ids_only/serial_recall/result_20260511_025bec41d7d94b669dabfbf4e2173ecd_zillizcloud.json) | 78.3911 | 83.4967 | [83.4967](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260511_7dbb463deb6244a8b065324e5bd1dd23_zillizcloud.json) | 0.7603s / 0.9511s | 0.8941s / 1.1150s | 1.0085s / 1.3020s | 2,000 | measured |
| 5% | scalar label | [0.9637](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/5p/scalar_label/serial_recall/result_20260511_807358732bf64658b2c0c9fdeb19afab_zillizcloud.json) | 74.2037 | 79.9345 | [79.9345](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260511_eedb3a0041584c108b1b8a95130f03f8_zillizcloud.json) | 0.7841s / 0.9948s | 0.9752s / 1.1840s | 1.0608s / 1.3049s | 3,600 | measured |
| 5% | vector | [0.9637](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/5p/vector/serial_recall/result_20260511_beb8df431cec477d96026a1bfd8abd18_zillizcloud.json) | 45.3072 | 49.9419 | [49.9419](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/5p/vector/concurrent_qps/result_20260511_b5f825e419a14323866ef10c82a269ab_zillizcloud.json) | 1.3171s / 1.5892s | 1.6811s / 1.8899s | 2.0103s / 2.1834s | 309,200 | measured |
| 10% | IDs only | [0.9607](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/10p/ids_only/serial_recall/result_20260511_6b15cb442c2147d393e0006eaf3e1b07_zillizcloud.json) | 44.6276 | 48.9215 | [48.9215](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260511_3daccea2972848be986c21c0dcf822af_zillizcloud.json) | 1.3387s / 1.6264s | 1.5379s / 2.1119s | 1.8242s / 2.3556s | 2,000 | measured |
| 10% | scalar label | [0.9607](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/10p/scalar_label/serial_recall/result_20260511_cccf0aed8a124d9c9c54c2ea191de3ba_zillizcloud.json) | 41.0114 | 42.0981 | [42.0981](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260511_b2d38defc7c74e508698e3c47e7cc853_zillizcloud.json) | 1.4503s / 1.8761s | 1.7349s / 2.2248s | 1.9109s / 2.3956s | 3,600 | measured |
| 10% | vector | [0.9607](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/10p/vector/serial_recall/result_20260511_8c5160bebdd84ab9958f4ce363f2b812_zillizcloud.json) | 24.8381 | 28.2022 | [28.2022](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/10p/vector/concurrent_qps/result_20260511_d806e624d11b49d2b4ba170d483a4d04_zillizcloud.json) | 2.3966s / 2.8057s | 2.8363s / 3.3085s | 3.1741s / 3.8916s | 309,200 | measured |
| 20% | IDs only | [0.9586](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/20p/ids_only/serial_recall/result_20260511_f25043323e744de5807128e6a9a03ed2_zillizcloud.json) | 30.6455 | 33.7269 | [33.7269](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260511_4ebc614f575e4573ab9f4404dc40d544_zillizcloud.json) | 1.9432s / 2.3374s | 2.1765s / 2.6635s | 2.2117s / 2.9054s | 2,000 | measured |
| 20% | scalar label | [0.9586](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/20p/scalar_label/serial_recall/result_20260511_a021c5d2083a407e826ca1cc4905a8aa_zillizcloud.json) | 29.9014 | 31.1327 | [31.1327](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260511_cc3d108862304a4584185eac6198227e_zillizcloud.json) | 1.9908s / 2.5375s | 2.3567s / 3.0346s | 2.5183s / 3.7292s | 3,600 | measured |
| 20% | vector | [0.9586](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/20p/vector/serial_recall/result_20260511_4d53363df031429ca672450312d75be4_zillizcloud.json) | 20.9791 | 22.5249 | [22.5249](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/20p/vector/concurrent_qps/result_20260511_42901e633eff44f6a6ebaaa31c16c28e_zillizcloud.json) | 2.8274s / 3.5225s | 3.5156s / 4.2164s | 3.9773s / 4.4819s | 309,200 | measured |
| 50% | IDs only | [0.9551](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/50p/ids_only/serial_recall/result_20260511_7dff183d893e40dfa10d3f1bafc361ff_zillizcloud.json) | 21.0573 | 22.7089 | [22.7089](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260511_007f6d744fef4368b15d9a2bfe9c4954_zillizcloud.json) | 2.8060s / 3.4676s | 3.1068s / 3.7737s | 3.1576s / 3.8198s | 2,000 | measured |
| 50% | scalar label | [0.9551](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/50p/scalar_label/serial_recall/result_20260511_b1a564a2b4c94448ba1b9a4d3722feac_zillizcloud.json) | 19.7032 | 21.4209 | [21.4209](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260511_aaefdab4fa174d95b19abcb2d1b14ab3_zillizcloud.json) | 3.0089s / 3.6526s | 3.4295s / 4.2006s | 3.6849s / 4.3236s | 3,600 | measured |
| 50% | vector | [0.9551](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/50p/vector/serial_recall/result_20260511_5d0e671cfe71462b8d3b05e50405bf0d_zillizcloud.json) | 15.9935 | 16.9123 | [16.9123](raw_results/zilliz_cloud_tiered_4cu/scalar_label_filter/50p/vector/concurrent_qps/result_20260511_310db7d28a5e4f19b1f8fe76fdb94b09_zillizcloud.json) | 3.6905s / 4.6920s | 4.2006s / 5.3905s | 5.1222s / 6.2679s | 309,200 | measured |

## Zilliz Cloud Capacity 12CU

| Item | Value |
|---|---|
| Collection | `LAION100M_capacity` |
| Scalar label field | `label` |
| Scalar label index | `labels_idx` |
| Logical row count | 100,000,000 |

### Unfiltered Search

| Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| IDs only | [0.9723](raw_results/zilliz_cloud_capacity_12cu/unfiltered/na/ids_only/serial_recall/result_20260511_17d87c351af745ceab0738d3155af9ac_zillizcloud.json) | 366.5669 | 376.0070 | [376.0070](raw_results/zilliz_cloud_capacity_12cu/unfiltered/na/ids_only/concurrent_qps/result_20260511_8ee5908066ca48f8848b56f01285c6fb_zillizcloud.json) | 0.1626s / 0.2108s | 0.1940s / 0.2748s | 0.2097s / 0.2993s | measured |
| scalar label | [0.9723](raw_results/zilliz_cloud_capacity_12cu/unfiltered/na/scalar_label/serial_recall/result_20260511_220657bd87e148ad95531a7601d23253_zillizcloud.json) | 379.4628 | 370.1937 | [379.4628](raw_results/zilliz_cloud_capacity_12cu/unfiltered/na/scalar_label/concurrent_qps/result_20260511_3ed9191186a84f0d83f5a15b98f1d42e_zillizcloud.json) | 0.1571s / 0.2141s | 0.1940s / 0.2739s | 0.2043s / 0.2929s | measured |
| vector | [0.9723](raw_results/zilliz_cloud_capacity_12cu/unfiltered/na/vector/serial_recall/result_20260511_a7261239b0234060be1d349b9174d4f8_zillizcloud.json) | 219.3919 | 229.4362 | [229.4362](raw_results/zilliz_cloud_capacity_12cu/unfiltered/na/vector/concurrent_qps/result_20260511_259a2d53626a4011bbde5c1cf0d7f82d_zillizcloud.json) | 0.2719s / 0.3455s | 0.3412s / 0.4440s | 0.3889s / 0.4977s | measured |

### Integer Filtered Search

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 99.9% | IDs only | [0.9781](raw_results/zilliz_cloud_capacity_12cu/int_filter/0_1p/ids_only/serial_recall/result_20260511_c935770a90e7408c90528fa1bd77b672_zillizcloud.json) | 1210.8837 | 1380.2318 | [1380.2318](raw_results/zilliz_cloud_capacity_12cu/int_filter/0_1p/ids_only/concurrent_qps/result_20260511_cfb93e52a67d480f8c435f6ed76d85e0_zillizcloud.json) | 0.0492s / 0.0574s | 0.0839s / 0.0873s | 0.0912s / 0.0921s | measured |
| 99.9% | scalar label | [0.9781](raw_results/zilliz_cloud_capacity_12cu/int_filter/0_1p/scalar_label/serial_recall/result_20260511_77bff551f0e44f958e3dce5026b461c0_zillizcloud.json) | 1183.8423 | 1338.3295 | [1338.3295](raw_results/zilliz_cloud_capacity_12cu/int_filter/0_1p/scalar_label/concurrent_qps/result_20260511_f32a550f1836499aa466c467b0d1b30d_zillizcloud.json) | 0.0504s / 0.0592s | 0.0826s / 0.0871s | 0.0880s / 0.0926s | measured |
| 99.9% | vector | [0.9781](raw_results/zilliz_cloud_capacity_12cu/int_filter/0_1p/vector/serial_recall/result_20260511_81b48162610c47ecbab8d5984a1c863c_zillizcloud.json) | 861.6848 | 933.0460 | [933.0460](raw_results/zilliz_cloud_capacity_12cu/int_filter/0_1p/vector/concurrent_qps/result_20260511_1d8cddb8d8c2460d9c74fe031cf52223_zillizcloud.json) | 0.0691s / 0.0848s | 0.0975s / 0.1086s | 0.1061s / 0.1182s | measured |
| 99% | IDs only | [0.9809](raw_results/zilliz_cloud_capacity_12cu/int_filter/1p/ids_only/serial_recall/result_20260511_905cb61feb424875910fec08da5bcd5f_zillizcloud.json) | 730.0303 | 792.0219 | [792.0219](raw_results/zilliz_cloud_capacity_12cu/int_filter/1p/ids_only/concurrent_qps/result_20260511_566d53410fdf4a4c8e150cf52f108f22_zillizcloud.json) | 0.0816s / 0.0999s | 0.1026s / 0.1637s | 0.1081s / 0.1702s | measured |
| 99% | scalar label | [0.9809](raw_results/zilliz_cloud_capacity_12cu/int_filter/1p/scalar_label/serial_recall/result_20260511_0c02402812de4d5eaee14ba795a552d0_zillizcloud.json) | 719.5956 | 785.7993 | [785.7993](raw_results/zilliz_cloud_capacity_12cu/int_filter/1p/scalar_label/concurrent_qps/result_20260511_2f29f98a1dad4130bdcf3e7c00e1eee9_zillizcloud.json) | 0.0828s / 0.1008s | 0.1031s / 0.1635s | 0.1090s / 0.1718s | measured |
| 99% | vector | [0.9809](raw_results/zilliz_cloud_capacity_12cu/int_filter/1p/vector/serial_recall/result_20260511_13c0324af6514d31a670c811da3436fb_zillizcloud.json) | 542.7760 | 572.6737 | [572.6737](raw_results/zilliz_cloud_capacity_12cu/int_filter/1p/vector/concurrent_qps/result_20260511_60b35e08ec914ef187a0818c5ec73c1a_zillizcloud.json) | 0.1098s / 0.1382s | 0.1664s / 0.1869s | 0.1764s / 0.1953s | measured |
| 90% | IDs only | [0.9852](raw_results/zilliz_cloud_capacity_12cu/int_filter/10p/ids_only/serial_recall/result_20260511_e130c430a3414dbcb5912636f3ffb91f_zillizcloud.json) | 717.1534 | 820.8570 | [820.8570](raw_results/zilliz_cloud_capacity_12cu/int_filter/10p/ids_only/concurrent_qps/result_20260511_dc8405bde48d4779a9027437bb77dcd8_zillizcloud.json) | 0.0831s / 0.0965s | 0.1012s / 0.1156s | 0.1055s / 0.1635s | measured |
| 90% | scalar label | [0.9852](raw_results/zilliz_cloud_capacity_12cu/int_filter/10p/scalar_label/serial_recall/result_20260511_7bb497b835d8456491d6fe9e585f00a4_zillizcloud.json) | 701.5481 | 756.5890 | [756.5890](raw_results/zilliz_cloud_capacity_12cu/int_filter/10p/scalar_label/concurrent_qps/result_20260511_530f75bd4831409aa6338c2726b1b70a_zillizcloud.json) | 0.0849s / 0.1046s | 0.1029s / 0.1619s | 0.1078s / 0.1706s | measured |
| 90% | vector | [0.9852](raw_results/zilliz_cloud_capacity_12cu/int_filter/10p/vector/serial_recall/result_20260511_cc9cbc49618b4c0dab8336c04e788111_zillizcloud.json) | 488.2515 | 517.2106 | [517.2106](raw_results/zilliz_cloud_capacity_12cu/int_filter/10p/vector/concurrent_qps/result_20260511_676a35c5b8644afaaf31e4865fe4f173_zillizcloud.json) | 0.1220s / 0.1530s | 0.1738s / 0.1939s | 0.1839s / 0.2037s | measured |
| 50% | IDs only | [0.9838](raw_results/zilliz_cloud_capacity_12cu/int_filter/50p/ids_only/serial_recall/result_20260511_9c7de03e08064f0aa58fa9c1aa899818_zillizcloud.json) | 322.8739 | 337.8539 | [337.8539](raw_results/zilliz_cloud_capacity_12cu/int_filter/50p/ids_only/concurrent_qps/result_20260511_a2630a165b35455aaa7090b589a4b865_zillizcloud.json) | 0.1847s / 0.2346s | 0.2065s / 0.2837s | 0.2468s / 0.2978s | measured |
| 50% | scalar label | [0.9838](raw_results/zilliz_cloud_capacity_12cu/int_filter/50p/scalar_label/serial_recall/result_20260511_a75369c0f8284e8ebb3aa4612fbd6101_zillizcloud.json) | 321.8107 | 320.3212 | [321.8107](raw_results/zilliz_cloud_capacity_12cu/int_filter/50p/scalar_label/concurrent_qps/result_20260511_a9cee107c9874ca98126c4a95907ef53_zillizcloud.json) | 0.1852s / 0.2476s | 0.2100s / 0.2917s | 0.2674s / 0.3212s | measured |
| 50% | vector | [0.9838](raw_results/zilliz_cloud_capacity_12cu/int_filter/50p/vector/serial_recall/result_20260511_d186a89459934d4087d1465c35517db1_zillizcloud.json) | 225.2714 | 237.8277 | [237.8277](raw_results/zilliz_cloud_capacity_12cu/int_filter/50p/vector/concurrent_qps/result_20260511_171bf6d177aa469dac317f1780c7faa8_zillizcloud.json) | 0.2647s / 0.3333s | 0.3156s / 0.4080s | 0.3818s / 0.4796s | measured |

### Scalar Label Filtered Search

1% scalar-label filter rate means `label == "label_1p"`, approximately 1M matched rows.

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---|
| 0.1% | IDs only | [0.9961](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_1p/ids_only/serial_recall/result_20260511_7aebda8f83234581a2072876182b71fd_zillizcloud.json) | 316.7608 | 316.7563 | [316.7608](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260511_f08983fa407149569a6aa3d977d93a9d_zillizcloud.json) | 0.1881s / 0.2503s | 0.2129s / 0.2827s | 0.2467s / 0.3607s | 2,000 | measured |
| 0.1% | scalar label | [0.9961](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_1p/scalar_label/serial_recall/result_20260511_a70e129539844d66b920b52bbf741610_zillizcloud.json) | 316.5333 | 316.6797 | [316.6797](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260511_979e04175e42419da8a7604f73019bf5_zillizcloud.json) | 0.1884s / 0.2503s | 0.2182s / 0.2884s | 0.2478s / 0.3115s | 3,600 | measured |
| 0.1% | vector | [0.9961](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_1p/vector/serial_recall/result_20260511_01d9606419ab4e31bcae85b04de67198_zillizcloud.json) | 228.3997 | 237.7134 | [237.7134](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260511_477b61925e624e6e9f0c5645be490429_zillizcloud.json) | 0.2610s / 0.3333s | 0.3317s / 0.4301s | 0.3870s / 0.4869s | 309,200 | measured |
| 0.2% | IDs only | [0.9957](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_2p/ids_only/serial_recall/result_20260511_f05dadd6d5b6457d9691ec6e9c076edf_zillizcloud.json) | 303.5832 | 316.3100 | [316.3100](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260511_494ee33cd07e4d7faac1d8648abcbe4c_zillizcloud.json) | 0.1964s / 0.2507s | 0.2236s / 0.2877s | 0.2571s / 0.3357s | 2,000 | measured |
| 0.2% | scalar label | [0.9957](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_2p/scalar_label/serial_recall/result_20260511_d16edb0d28e9481a9a47bffc4a22401d_zillizcloud.json) | 290.1460 | 317.4258 | [317.4258](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260511_44e765c33aa840c891e78632faf78360_zillizcloud.json) | 0.2055s / 0.2496s | 0.2367s / 0.2899s | 0.2572s / 0.3212s | 3,600 | measured |
| 0.2% | vector | [0.9957](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_2p/vector/serial_recall/result_20260511_6a95b1e86f1844f291e22c029aced700_zillizcloud.json) | 206.5308 | 203.0653 | [206.5308](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260511_6a6a42a5de364ccba4f878b4d2487778_zillizcloud.json) | 0.2887s / 0.3902s | 0.3722s / 0.4919s | 0.4151s / 0.5568s | 309,200 | measured |
| 0.5% | IDs only | [0.9955](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_5p/ids_only/serial_recall/result_20260511_362836dcab534b7d816aa5815330f663_zillizcloud.json) | 205.0275 | 209.1240 | [209.1240](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260511_9a8478cd09cb4613a0b7c8a954b97ff8_zillizcloud.json) | 0.2908s / 0.3794s | 0.3516s / 0.4547s | 0.3966s / 0.5014s | 2,000 | measured |
| 0.5% | scalar label | [0.9955](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_5p/scalar_label/serial_recall/result_20260511_955396a103274166ad2c073e5052c81b_zillizcloud.json) | 200.0110 | 208.3996 | [208.3996](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260511_cefb61cd9ce54b22a331b22808d03e0b_zillizcloud.json) | 0.2982s / 0.3810s | 0.3636s / 0.4526s | 0.3914s / 0.4807s | 3,600 | measured |
| 0.5% | vector | [0.9955](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_5p/vector/serial_recall/result_20260511_fb924ab82f6c4cd1b2b8426d0dfa463d_zillizcloud.json) | 153.4133 | 153.5717 | [153.5717](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260511_8c2bcfe3f60a40bbb2cba1e89bd33072_zillizcloud.json) | 0.3890s / 0.5175s | 0.4837s / 0.6282s | 0.5293s / 0.7104s | 309,200 | measured |
| 1% | IDs only | [0.9951](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/1p/ids_only/serial_recall/result_20260511_c1863f581a8742e9ba4a408e81cabed5_zillizcloud.json) | 136.5733 | 136.1872 | [136.5733](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260511_6fdd2c40f5944cf3a9a67048ddfbbce0_zillizcloud.json) | 0.4370s / 0.5824s | 0.5071s / 0.6842s | 0.6275s / 0.7718s | 2,000 | measured |
| 1% | scalar label | [0.9951](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/1p/scalar_label/serial_recall/result_20260511_770580bd27134d80b4ab4b3382854900_zillizcloud.json) | 133.7402 | 138.1391 | [138.1391](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260511_7ef25bda77d444f78d4421e6312bae30_zillizcloud.json) | 0.4455s / 0.5747s | 0.5132s / 0.6883s | 0.5885s / 0.7135s | 3,600 | measured |
| 1% | vector | [0.9951](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/1p/vector/serial_recall/result_20260511_6f300afddb40448887ddc7ea2ddfedf2_zillizcloud.json) | 111.3446 | 110.6730 | [111.3446](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/1p/vector/concurrent_qps/result_20260511_2bdcdd6cc63145f39f720d7d7f929f4b_zillizcloud.json) | 0.5354s / 0.7170s | 0.6691s / 0.8759s | 0.7793s / 0.9838s | 309,200 | measured |
| 2% | IDs only | [0.9914](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/2p/ids_only/serial_recall/result_20260511_b217a4c2b8ec442aa51cf9b43a38ce36_zillizcloud.json) | 182.2757 | 163.7759 | [182.2757](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260511_2b3039350a6d429ca19753e266feaa59_zillizcloud.json) | 0.3272s / 0.4844s | 0.3852s / 0.5866s | 0.4455s / 0.6584s | 2,000 | measured |
| 2% | scalar label | [0.9914](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/2p/scalar_label/serial_recall/result_20260511_4c0357e64f53437d96cc1c2a2019f6fb_zillizcloud.json) | 155.4340 | 160.1895 | [160.1895](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260511_345a83914a134dba89243f63a49c2d82_zillizcloud.json) | 0.3841s / 0.4953s | 0.4712s / 0.5870s | 0.5047s / 0.6309s | 3,600 | measured |
| 2% | vector | [0.9914](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/2p/vector/serial_recall/result_20260511_980219a6e8d84f09ba8ff36e6edf1bfe_zillizcloud.json) | 140.8582 | 127.8600 | [140.8582](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/2p/vector/concurrent_qps/result_20260511_79efcf0d2fa64741ae0737954772db6e_zillizcloud.json) | 0.4236s / 0.6202s | 0.5193s / 0.7716s | 0.6007s / 0.8662s | 309,200 | measured |
| 5% | IDs only | [0.9912](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/5p/ids_only/serial_recall/result_20260511_5510a12c91234476978651663bb83829_zillizcloud.json) | 116.1035 | 127.1201 | [127.1201](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260511_9dae502a1aa94bd4b3c881a6de754009_zillizcloud.json) | 0.5136s / 0.6232s | 0.6000s / 0.7739s | 0.6927s / 0.9193s | 2,000 | measured |
| 5% | scalar label | [0.9912](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/5p/scalar_label/serial_recall/result_20260511_cba87e72facb459c8e32404126607c6e_zillizcloud.json) | 134.0880 | 143.3337 | [143.3337](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260511_274b64ccd0144452808e05bae55beaf7_zillizcloud.json) | 0.4451s / 0.5532s | 0.5046s / 0.6919s | 0.5826s / 0.7248s | 3,600 | measured |
| 5% | vector | [0.9912](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/5p/vector/serial_recall/result_20260511_d68bf1acf4f744729d9cd40e50f197c0_zillizcloud.json) | 114.3365 | 122.0458 | [122.0458](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/5p/vector/concurrent_qps/result_20260511_07cebf82c48c441badb5e6545d5337c8_zillizcloud.json) | 0.5223s / 0.6505s | 0.6559s / 0.8066s | 0.7297s / 0.9092s | 309,200 | measured |
| 10% | IDs only | [0.9897](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/10p/ids_only/serial_recall/result_20260511_df1e16c7409c467db630464e34c88f2a_zillizcloud.json) | 108.6286 | 105.7116 | [108.6286](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260511_72844b316698430bb9b457a92091b979_zillizcloud.json) | 0.5502s / 0.7505s | 0.6626s / 0.8850s | 0.7860s / 1.0712s | 2,000 | measured |
| 10% | scalar label | [0.9897](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/10p/scalar_label/serial_recall/result_20260511_68ba5f9c4aaa4844815062289ca83419_zillizcloud.json) | 101.8603 | 105.5498 | [105.5498](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260511_43b44c9c65464fe7947ad921c7dbf39a_zillizcloud.json) | 0.5863s / 0.7517s | 0.6808s / 0.8833s | 0.7151s / 0.9339s | 3,600 | measured |
| 10% | vector | [0.9897](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/10p/vector/serial_recall/result_20260511_ba29911643e4455cbda45eaf38becaf4_zillizcloud.json) | 93.9811 | 90.9934 | [93.9811](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/10p/vector/concurrent_qps/result_20260511_2e16902d65064fea9052736ca247b6c8_zillizcloud.json) | 0.6362s / 0.8738s | 0.7931s / 1.0591s | 0.9160s / 1.1957s | 309,200 | measured |
| 20% | IDs only | [0.9892](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/20p/ids_only/serial_recall/result_20260511_b6c513276f5f4d2391064da0ed241241_zillizcloud.json) | 87.6680 | 86.6352 | [87.6680](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260511_61362bab8ac24b3abb0fac4772200a0d_zillizcloud.json) | 0.6800s / 0.9147s | 0.7766s / 1.0480s | 0.7885s / 1.1652s | 2,000 | measured |
| 20% | scalar label | [0.9892](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/20p/scalar_label/serial_recall/result_20260511_4e5e68f8640a41b1a1e0e520216f2380_zillizcloud.json) | 78.9013 | 86.8571 | [86.8571](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260511_8f7d841b174144acb59f3520d95e641c_zillizcloud.json) | 0.7561s / 0.9131s | 0.8722s / 1.0662s | 0.9246s / 1.1248s | 3,600 | measured |
| 20% | vector | [0.9892](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/20p/vector/serial_recall/result_20260511_4ea6d3594b2d44d1b9d722c2d15a156c_zillizcloud.json) | 76.5511 | 77.9215 | [77.9215](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/20p/vector/concurrent_qps/result_20260511_e3a44ef336c847bca73b2846d58466e5_zillizcloud.json) | 0.7805s / 1.0204s | 0.9585s / 1.2087s | 1.1737s / 1.3802s | 309,200 | measured |
| 50% | IDs only | [0.9847](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/50p/ids_only/serial_recall/result_20260511_e4f5b746fff54cb9ae75b8a10fe34e1b_zillizcloud.json) | 95.6923 | 109.4589 | [109.4589](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260511_5be830c78b2244aba70440743cab58e8_zillizcloud.json) | 0.6219s / 0.7244s | 0.6874s / 0.7955s | 0.7768s / 0.8118s | 2,000 | measured |
| 50% | scalar label | [0.9847](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/50p/scalar_label/serial_recall/result_20260511_620d70bf55114dadae8fb45ae4614d33_zillizcloud.json) | 96.6743 | 113.3363 | [113.3363](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260511_adc481db9bde49dcb011f5053e726662_zillizcloud.json) | 0.6165s / 0.6991s | 0.6937s / 0.7911s | 0.7678s / 0.8935s | 3,600 | measured |
| 50% | vector | [0.9847](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/50p/vector/serial_recall/result_20260511_6a17585a0ae84f759d02280c20821c0b_zillizcloud.json) | 88.6614 | 96.9616 | [96.9616](raw_results/zilliz_cloud_capacity_12cu/scalar_label_filter/50p/vector/concurrent_qps/result_20260511_71839a0964b4498a87a11072679711b3_zillizcloud.json) | 0.6722s / 0.8183s | 0.7954s / 0.9553s | 0.9112s / 1.1676s | 309,200 | measured |

## Pinecone Serverless

Index layout and CloudPayloadSearchCase compatibility validated on
`vdbbench-laion-100m-768d-l2` with 100,000,000 vectors, 768 dimensions, and L2
metric. Metadata fields used for payload tests: `label`, `label_2kb`, and
`meta`.

Pinecone unfiltered concurrent search was run at concurrency `3,4` with
VDBBench Pinecone query retry/backoff enabled. These concurrency levels are
lower than the default `60,80` because the serverless index returned 429
rate-limit responses at higher unpaced concurrency.

### Unfiltered Search

| Payload | Recall | QPS @3 | QPS @4 | Max QPS | Avg latency @3/@4 | P95 @3/@4 | P99 @3/@4 | Payload bytes/query | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| IDs only | [0.9609](raw_results/pinecone_serverless/unfiltered/na/ids_only/serial_recall/result_20260513_pinecone_serverless_unfiltered_na_ids_only_serial_recall_20260513_pinecone.json) | 4.5642 | 4.1479 | [4.5642](raw_results/pinecone_serverless/unfiltered/na/ids_only/concurrent_qps/result_20260513_pinecone_serverless_unfiltered_na_ids_only_concurrent_qps_c3c4_20260513_pinecone.json) | 0.6546s / 0.9537s | 2.0774s / 3.0967s | 2.6046s / 4.8496s | 2,000 | measured retry-enabled |
| scalar label | [0.9609](raw_results/pinecone_serverless/unfiltered/na/scalar_label/serial_recall/result_20260513_pinecone_serverless_unfiltered_na_scalar_label_serial_recall_20260513_pinecone.json) | 4.5139 | 4.1723 | [4.5139](raw_results/pinecone_serverless/unfiltered/na/scalar_label/concurrent_qps/result_20260513_pinecone_serverless_unfiltered_na_scalar_label_concurrent_qps_c3c4_20260513_pinecone.json) | 0.6592s / 0.9486s | 2.0939s / 2.6068s | 2.6436s / 4.3413s | 3,600 | measured retry-enabled |
| vector | [0.961](raw_results/pinecone_serverless/unfiltered/na/vector/serial_recall/result_20260513_pinecone_serverless_unfiltered_na_vector_serial_recall_20260513_pinecone.json) | 4.4830 | 4.1931 | [4.483](raw_results/pinecone_serverless/unfiltered/na/vector/concurrent_qps/result_20260513_pinecone_serverless_unfiltered_na_vector_concurrent_qps_c3c4_20260513_pinecone.json) | 0.6623s / 0.9412s | 1.0895s / 2.0901s | 1.6160s / 2.6097s | 309,200 | measured retry-enabled |

### Integer Filtered Search

| Filter rate | Payload | Recall | QPS @4 | Max QPS | Avg latency @4 | P95 @4 | P99 @4 | Status |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 99.9% | IDs only | pending | 4.7080 | [4.7080](raw_results/pinecone_serverless/int_filter/0_1p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_int_filter_0_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8336s | 2.0966s | 2.6222s | measured QPS @4; recall pending |
| 99.9% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | pending |
| 99.9% | vector | pending | 4.7639 | [4.7639](raw_results/pinecone_serverless/int_filter/0_1p/vector/concurrent_qps/result_20260514_pinecone_serverless_int_filter_0_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8121s | 2.0074s | 3.2960s | measured QPS @4; recall pending |
| 99% | IDs only | pending | 5.1368 | [5.1368](raw_results/pinecone_serverless/int_filter/1p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_int_filter_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7622s | 2.1487s | 2.9381s | measured QPS @4; recall pending |
| 99% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | pending |
| 99% | vector | pending | 4.7374 | [4.7374](raw_results/pinecone_serverless/int_filter/1p/vector/concurrent_qps/result_20260514_pinecone_serverless_int_filter_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8262s | 2.0890s | 2.6605s | measured QPS @4; recall pending |
| 90% | IDs only | pending | 4.7933 | [4.7933](raw_results/pinecone_serverless/int_filter/10p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_int_filter_10p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8208s | 2.2307s | 3.6887s | measured QPS @4; recall pending |
| 90% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | pending |
| 90% | vector | pending | 5.1852 | [5.1852](raw_results/pinecone_serverless/int_filter/10p/vector/concurrent_qps/result_20260514_pinecone_serverless_int_filter_10p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7630s | 0.9579s | 1.0734s | measured QPS @4; recall pending |
| 50% | IDs only | pending | 4.6981 | [4.6981](raw_results/pinecone_serverless/int_filter/50p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_int_filter_50p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8255s | 2.4822s | 3.2012s | measured QPS @4; recall pending |
| 50% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | pending |
| 50% | vector | pending | 5.1066 | [5.1066](raw_results/pinecone_serverless/int_filter/50p/vector/concurrent_qps/result_20260514_pinecone_serverless_int_filter_50p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7607s | 1.1826s | 1.6863s | measured QPS @4; recall pending |

### Scalar Label Filtered Search

| Filter rate | Payload | Recall | QPS @4 | Max QPS | Avg latency @4 | P95 @4 | P99 @4 | Payload bytes/query | Status |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 0.1% | IDs only | pending | 4.7633 | [4.7633](raw_results/pinecone_serverless/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_0_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7754s | 2.6086s | 3.3792s | 2,000 | measured QPS @4; recall pending |
| 0.1% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 0.1% | vector | pending | 4.7593 | [4.7593](raw_results/pinecone_serverless/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_0_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8223s | 1.6412s | 2.1756s | 309,200 | measured QPS @4; recall pending |
| 0.2% | IDs only | pending | 4.8166 | [4.8166](raw_results/pinecone_serverless/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_0_2p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8102s | 2.6868s | 3.1605s | 2,000 | measured QPS @4; recall pending |
| 0.2% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 0.2% | vector | pending | 4.7140 | [4.7140](raw_results/pinecone_serverless/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_0_2p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8208s | 1.4816s | 2.0104s | 309,200 | measured QPS @4; recall pending |
| 0.5% | IDs only | pending | 4.7515 | [4.7515](raw_results/pinecone_serverless/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_0_5p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7839s | 2.2104s | 3.4686s | 2,000 | measured QPS @4; recall pending |
| 0.5% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 0.5% | vector | pending | 4.8229 | [4.8229](raw_results/pinecone_serverless/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_0_5p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8105s | 1.1027s | 1.2585s | 309,200 | measured QPS @4; recall pending |
| 1% | IDs only | pending | 4.8065 | [4.8065](raw_results/pinecone_serverless/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_1p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8183s | 2.2159s | 2.8347s | 2,000 | measured QPS @4; recall pending |
| 1% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 1% | vector | pending | 4.0442 | [4.0442](raw_results/pinecone_serverless/scalar_label_filter/1p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_1p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.9775s | 1.4655s | 1.6721s | 309,200 | measured QPS @4; recall pending |
| 2% | IDs only | pending | 4.5753 | [4.5753](raw_results/pinecone_serverless/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_2p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8658s | 1.1790s | 1.2997s | 2,000 | measured QPS @4; recall pending |
| 2% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 2% | vector | pending | 3.6870 | [3.6870](raw_results/pinecone_serverless/scalar_label_filter/2p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_2p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 1.0700s | 1.2998s | 1.5606s | 309,200 | measured QPS @4; recall pending |
| 5% | IDs only | pending | 3.2915 | [3.2915](raw_results/pinecone_serverless/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_5p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 1.2012s | 1.7524s | 1.9216s | 2,000 | measured QPS @4; recall pending |
| 5% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 5% | vector | pending | 2.8416 | [2.8416](raw_results/pinecone_serverless/scalar_label_filter/5p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_5p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 1.3851s | 2.0146s | 2.2025s | 309,200 | measured QPS @4; recall pending |
| 10% | IDs only | pending | 4.1620 | [4.1620](raw_results/pinecone_serverless/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_10p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.9470s | 1.3166s | 1.5917s | 2,000 | measured QPS @4; recall pending |
| 10% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 10% | vector | pending | 3.3833 | [3.3833](raw_results/pinecone_serverless/scalar_label_filter/10p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_10p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 1.1624s | 1.4733s | 1.6251s | 309,200 | measured QPS @4; recall pending |
| 20% | IDs only | pending | 5.2239 | [5.2239](raw_results/pinecone_serverless/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_20p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7499s | 2.6231s | 3.3569s | 2,000 | measured QPS @4; recall pending |
| 20% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 20% | vector | pending | 5.1222 | [5.1222](raw_results/pinecone_serverless/scalar_label_filter/20p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_20p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.7680s | 1.0751s | 1.2313s | 309,200 | measured QPS @4; recall pending |
| 50% | IDs only | pending | 4.7458 | [4.7458](raw_results/pinecone_serverless/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_50p_ids_only_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8169s | 2.6630s | 3.4490s | 2,000 | measured QPS @4; recall pending |
| 50% | scalar label | pending | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| 50% | vector | pending | 4.7549 | [4.7549](raw_results/pinecone_serverless/scalar_label_filter/50p/vector/concurrent_qps/result_20260514_pinecone_serverless_scalar_label_filter_50p_vector_concurrent_qps_c4_30s_20260514_pinecone.json) | 0.8245s | 1.5453s | 1.8657s | 309,200 | measured QPS @4; recall pending |

## Turbopuffer Unpinned

These results are for the unpinned `laion100m_bulk` namespace.

| Item | Value |
|---|---|
| Namespace | `laion100m_bulk` |
| Scalar label field | `label` |
| Logical row count | 100,000,000 |
| Concurrency | `60,80` |
| Duration note | Unfiltered and the first two 99.9% integer-filter concurrent runs used 60s because they completed before the duration switch; all remaining concurrent runs used 30s. Raw JSON files are the source of truth. |

### Unfiltered Search

| Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| IDs only | [0.9321](raw_results/turbopuffer_unpinned/unfiltered/na/ids_only/serial_recall/result_20260514_turbopuffer_unfiltered_na_ids_only_serial_recall_20260514_turbopuffer.json) | 243.7375 | 395.6972 | [395.6972](raw_results/turbopuffer_unpinned/unfiltered/na/ids_only/concurrent_qps/result_20260514_turbopuffer_unfiltered_na_ids_only_topk100_c60_c80_60s_20260514_turbopuffer.json) | 0.2423s / 0.1999s | 1.1186s / 1.0838s | 2.3392s / 2.2622s | measured |
| scalar label | [0.9321](raw_results/turbopuffer_unpinned/unfiltered/na/scalar_label/serial_recall/result_20260514_turbopuffer_unfiltered_na_scalar_label_serial_recall_20260514_turbopuffer.json) | 395.0372 | 399.6005 | [399.6005](raw_results/turbopuffer_unpinned/unfiltered/na/scalar_label/concurrent_qps/result_20260514_turbopuffer_unfiltered_na_scalar_label_topk100_c60_c80_60s_20260514_turbopuffer.json) | 0.1500s / 0.1974s | 1.0727s / 1.0863s | 1.1188s / 2.2783s | measured |
| vector | [0.9321](raw_results/turbopuffer_unpinned/unfiltered/na/vector/serial_recall/result_20260514_turbopuffer_unfiltered_na_vector_serial_recall_20260514_turbopuffer.json) | 367.7746 | 382.1502 | [382.1502](raw_results/turbopuffer_unpinned/unfiltered/na/vector/concurrent_qps/result_20260514_turbopuffer_unfiltered_na_vector_topk100_c60_c80_60s_20260514_turbopuffer.json) | 0.1615s / 0.2055s | 1.0834s / 1.0998s | 1.2491s / 2.3192s | measured |

### Integer Filtered Search

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 99.9% | IDs only | [0.9436](raw_results/turbopuffer_unpinned/int_filter/0_1p/ids_only/serial_recall/result_20260514_turbopuffer_int_filter_0_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 44.5217 | 44.3561 | [44.5217](raw_results/turbopuffer_unpinned/int_filter/0_1p/ids_only/concurrent_qps/result_20260514_turbopuffer_int_filter_0_1p_ids_only_topk100_c60_c80_60s_20260514_turbopuffer.json) | 1.1724s / 1.3184s | 4.3077s / 4.4042s | 6.9885s / 7.0134s | measured |
| 99.9% | scalar label | [0.9436](raw_results/turbopuffer_unpinned/int_filter/0_1p/scalar_label/serial_recall/result_20260514_turbopuffer_int_filter_0_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 46.0628 | 46.5236 | [46.5236](raw_results/turbopuffer_unpinned/int_filter/0_1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_int_filter_0_1p_scalar_label_topk100_c60_c80_60s_20260514_turbopuffer.json) | 1.1640s / 1.3469s | 4.2861s / 4.3630s | 6.9679s / 6.9950s | measured |
| 99.9% | vector | [0.9436](raw_results/turbopuffer_unpinned/int_filter/0_1p/vector/serial_recall/result_20260514_turbopuffer_int_filter_0_1p_vector_serial_recall_20260514_turbopuffer.json) | 45.0710 | 44.2466 | [45.0710](raw_results/turbopuffer_unpinned/int_filter/0_1p/vector/concurrent_qps/result_20260514_turbopuffer_int_filter_0_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.2295s / 1.4373s | 4.2600s / 4.3950s | 6.8704s / 7.0341s | measured |
| 99% | IDs only | [0.7119](raw_results/turbopuffer_unpinned/int_filter/1p/ids_only/serial_recall/result_20260514_turbopuffer_int_filter_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 231.8324 | 240.4486 | [240.4486](raw_results/turbopuffer_unpinned/int_filter/1p/ids_only/concurrent_qps/result_20260514_turbopuffer_int_filter_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2531s / 0.3229s | 1.1277s / 1.1372s | 2.3511s / 2.3824s | measured |
| 99% | scalar label | [0.7119](raw_results/turbopuffer_unpinned/int_filter/1p/scalar_label/serial_recall/result_20260514_turbopuffer_int_filter_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 232.6180 | 231.9649 | [232.6180](raw_results/turbopuffer_unpinned/int_filter/1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_int_filter_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2509s / 0.3358s | 1.1266s / 1.1432s | 2.3361s / 2.3916s | measured |
| 99% | vector | [0.7119](raw_results/turbopuffer_unpinned/int_filter/1p/vector/serial_recall/result_20260514_turbopuffer_int_filter_1p_vector_serial_recall_20260514_turbopuffer.json) | 215.9243 | 198.9953 | [215.9243](raw_results/turbopuffer_unpinned/int_filter/1p/vector/concurrent_qps/result_20260514_turbopuffer_int_filter_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2718s / 0.3769s | 1.1478s / 1.1800s | 2.3892s / 2.4438s | measured |
| 90% | IDs only | [0.9076](raw_results/turbopuffer_unpinned/int_filter/10p/ids_only/serial_recall/result_20260514_turbopuffer_int_filter_10p_ids_only_serial_recall_20260514_turbopuffer.json) | 214.7339 | 226.6961 | [226.6961](raw_results/turbopuffer_unpinned/int_filter/10p/ids_only/concurrent_qps/result_20260514_turbopuffer_int_filter_10p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2730s / 0.3421s | 1.1365s / 1.1463s | 2.3567s / 2.4010s | measured |
| 90% | scalar label | [0.9076](raw_results/turbopuffer_unpinned/int_filter/10p/scalar_label/serial_recall/result_20260514_turbopuffer_int_filter_10p_scalar_label_serial_recall_20260514_turbopuffer.json) | 221.8696 | 215.1473 | [221.8696](raw_results/turbopuffer_unpinned/int_filter/10p/scalar_label/concurrent_qps/result_20260514_turbopuffer_int_filter_10p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2656s / 0.3441s | 1.1347s / 1.1468s | 2.3595s / 2.4006s | measured |
| 90% | vector | [0.9076](raw_results/turbopuffer_unpinned/int_filter/10p/vector/serial_recall/result_20260514_turbopuffer_int_filter_10p_vector_serial_recall_20260514_turbopuffer.json) | 211.9829 | 214.1514 | [214.1514](raw_results/turbopuffer_unpinned/int_filter/10p/vector/concurrent_qps/result_20260514_turbopuffer_int_filter_10p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2774s / 0.3627s | 1.1492s / 1.1658s | 2.4011s / 2.4378s | measured |
| 50% | IDs only | [0.9507](raw_results/turbopuffer_unpinned/int_filter/50p/ids_only/serial_recall/result_20260514_turbopuffer_int_filter_50p_ids_only_serial_recall_20260514_turbopuffer.json) | 207.6223 | 222.5367 | [222.5367](raw_results/turbopuffer_unpinned/int_filter/50p/ids_only/concurrent_qps/result_20260514_turbopuffer_int_filter_50p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2808s / 0.3450s | 1.1434s / 1.1488s | 2.3781s / 2.4151s | measured |
| 50% | scalar label | [0.9507](raw_results/turbopuffer_unpinned/int_filter/50p/scalar_label/serial_recall/result_20260514_turbopuffer_int_filter_50p_scalar_label_serial_recall_20260514_turbopuffer.json) | 223.5825 | 200.4144 | [223.5825](raw_results/turbopuffer_unpinned/int_filter/50p/scalar_label/concurrent_qps/result_20260514_turbopuffer_int_filter_50p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2612s / 0.3835s | 1.1309s / 1.1671s | 2.3625s / 2.4257s | measured |
| 50% | vector | [0.9507](raw_results/turbopuffer_unpinned/int_filter/50p/vector/serial_recall/result_20260514_turbopuffer_int_filter_50p_vector_serial_recall_20260514_turbopuffer.json) | 217.6923 | 204.0932 | [217.6923](raw_results/turbopuffer_unpinned/int_filter/50p/vector/concurrent_qps/result_20260514_turbopuffer_int_filter_50p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2690s / 0.3749s | 1.1467s / 1.1788s | 2.3865s / 2.4310s | measured |

### Scalar Label Filtered Search

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 0.1% | IDs only | [0.9438](raw_results/turbopuffer_unpinned/scalar_label_filter/0_1p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 56.4690 | 56.3511 | [56.4690](raw_results/turbopuffer_unpinned/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.9795s / 1.2451s | 2.6828s / 4.2907s | 6.5278s / 6.8447s | 2,000 | measured |
| 0.1% | scalar label | [0.9438](raw_results/turbopuffer_unpinned/scalar_label_filter/0_1p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 56.5172 | 54.4423 | [56.5172](raw_results/turbopuffer_unpinned/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.9961s / 1.2455s | 2.6856s / 4.3033s | 6.6828s / 6.8625s | 3,600 | measured |
| 0.1% | vector | [0.9438](raw_results/turbopuffer_unpinned/scalar_label_filter/0_1p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_1p_vector_serial_recall_20260514_turbopuffer.json) | 54.1251 | 54.2955 | [54.2955](raw_results/turbopuffer_unpinned/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.0211s / 1.2563s | 2.7137s / 4.3071s | 6.8135s / 6.9456s | 309,200 | measured |
| 0.2% | IDs only | [0.9375](raw_results/turbopuffer_unpinned/scalar_label_filter/0_2p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_2p_ids_only_serial_recall_20260514_turbopuffer.json) | 54.4188 | 55.5387 | [55.5387](raw_results/turbopuffer_unpinned/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_2p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.0001s / 1.2412s | 2.7334s / 4.3015s | 6.6530s / 6.8545s | 2,000 | measured |
| 0.2% | scalar label | [0.9375](raw_results/turbopuffer_unpinned/scalar_label_filter/0_2p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_2p_scalar_label_serial_recall_20260514_turbopuffer.json) | 56.2027 | 52.8577 | [56.2027](raw_results/turbopuffer_unpinned/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_2p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.9968s / 1.2709s | 2.7651s / 4.3113s | 6.8818s / 6.9120s | 3,600 | measured |
| 0.2% | vector | [0.9375](raw_results/turbopuffer_unpinned/scalar_label_filter/0_2p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_2p_vector_serial_recall_20260514_turbopuffer.json) | 51.5716 | 53.1789 | [53.1789](raw_results/turbopuffer_unpinned/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_2p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.0661s / 1.2706s | 3.2195s / 4.3045s | 6.6616s / 6.8330s | 309,200 | measured |
| 0.5% | IDs only | [0.5884](raw_results/turbopuffer_unpinned/scalar_label_filter/0_5p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_5p_ids_only_serial_recall_20260514_turbopuffer.json) | 260.7292 | 271.5802 | [271.5802](raw_results/turbopuffer_unpinned/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_5p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2249s / 0.2848s | 1.1143s / 1.1222s | 2.3188s / 2.3712s | 2,000 | measured |
| 0.5% | scalar label | [0.5884](raw_results/turbopuffer_unpinned/scalar_label_filter/0_5p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_5p_scalar_label_serial_recall_20260514_turbopuffer.json) | 267.9108 | 269.7651 | [269.7651](raw_results/turbopuffer_unpinned/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_5p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2197s / 0.2895s | 1.1125s / 1.1260s | 2.3155s / 2.3725s | 3,600 | measured |
| 0.5% | vector | [0.5884](raw_results/turbopuffer_unpinned/scalar_label_filter/0_5p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_0_5p_vector_serial_recall_20260514_turbopuffer.json) | 244.8822 | 234.9896 | [244.8822](raw_results/turbopuffer_unpinned/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_0_5p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2394s / 0.3172s | 1.1336s / 1.1480s | 2.3555s / 2.4145s | 309,200 | measured |
| 1% | IDs only | [0.7081](raw_results/turbopuffer_unpinned/scalar_label_filter/1p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 248.7513 | 258.3686 | [258.3686](raw_results/turbopuffer_unpinned/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2365s / 0.3008s | 1.1182s / 1.1269s | 2.3219s / 2.3845s | 2,000 | measured |
| 1% | scalar label | [0.7081](raw_results/turbopuffer_unpinned/scalar_label_filter/1p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 250.8213 | 249.1971 | [250.8213](raw_results/turbopuffer_unpinned/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2334s / 0.3125s | 1.1205s / 1.1328s | 2.3249s / 2.3888s | 3,600 | measured |
| 1% | vector | [0.7081](raw_results/turbopuffer_unpinned/scalar_label_filter/1p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_1p_vector_serial_recall_20260514_turbopuffer.json) | 242.0799 | 236.8867 | [242.0799](raw_results/turbopuffer_unpinned/scalar_label_filter/1p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2409s / 0.3276s | 1.1313s / 1.1543s | 2.3575s / 2.4178s | 309,200 | measured |
| 2% | IDs only | [0.7976](raw_results/turbopuffer_unpinned/scalar_label_filter/2p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_2p_ids_only_serial_recall_20260514_turbopuffer.json) | 226.9630 | 229.3709 | [229.3709](raw_results/turbopuffer_unpinned/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_2p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2542s / 0.3326s | 1.1277s / 1.1430s | 2.3502s / 2.3940s | 2,000 | measured |
| 2% | scalar label | [0.7976](raw_results/turbopuffer_unpinned/scalar_label_filter/2p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_2p_scalar_label_serial_recall_20260514_turbopuffer.json) | 237.9911 | 213.0843 | [237.9911](raw_results/turbopuffer_unpinned/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_2p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2462s / 0.3648s | 1.1234s / 1.1549s | 2.3534s / 2.4165s | 3,600 | measured |
| 2% | vector | [0.7976](raw_results/turbopuffer_unpinned/scalar_label_filter/2p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_2p_vector_serial_recall_20260514_turbopuffer.json) | 198.1984 | 220.5019 | [220.5019](raw_results/turbopuffer_unpinned/scalar_label_filter/2p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_2p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2840s / 0.3522s | 1.1517s / 1.1680s | 2.3805s / 2.4190s | 309,200 | measured |
| 5% | IDs only | [0.8746](raw_results/turbopuffer_unpinned/scalar_label_filter/5p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_5p_ids_only_serial_recall_20260514_turbopuffer.json) | 218.6052 | 224.8925 | [224.8925](raw_results/turbopuffer_unpinned/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_5p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2629s / 0.3388s | 1.1312s / 1.1493s | 2.3634s / 2.4020s | 2,000 | measured |
| 5% | scalar label | [0.8746](raw_results/turbopuffer_unpinned/scalar_label_filter/5p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_5p_scalar_label_serial_recall_20260514_turbopuffer.json) | 213.5266 | 224.9809 | [224.9809](raw_results/turbopuffer_unpinned/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_5p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2744s / 0.3454s | 1.1393s / 1.1499s | 2.3784s / 2.4051s | 3,600 | measured |
| 5% | vector | [0.8746](raw_results/turbopuffer_unpinned/scalar_label_filter/5p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_5p_vector_serial_recall_20260514_turbopuffer.json) | 203.2631 | 207.0767 | [207.0767](raw_results/turbopuffer_unpinned/scalar_label_filter/5p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_5p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2843s / 0.3681s | 1.1559s / 1.1760s | 2.3910s / 2.4527s | 309,200 | measured |
| 10% | IDs only | [0.9060](raw_results/turbopuffer_unpinned/scalar_label_filter/10p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_10p_ids_only_serial_recall_20260514_turbopuffer.json) | 202.6928 | 211.0448 | [211.0448](raw_results/turbopuffer_unpinned/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_10p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2908s / 0.3647s | 1.1482s / 1.1556s | 2.3867s / 2.4313s | 2,000 | measured |
| 10% | scalar label | [0.9060](raw_results/turbopuffer_unpinned/scalar_label_filter/10p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_10p_scalar_label_serial_recall_20260514_turbopuffer.json) | 224.4608 | 223.0433 | [224.4608](raw_results/turbopuffer_unpinned/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_10p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2598s / 0.3491s | 1.1301s / 1.1475s | 2.3677s / 2.4117s | 3,600 | measured |
| 10% | vector | [0.9060](raw_results/turbopuffer_unpinned/scalar_label_filter/10p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_10p_vector_serial_recall_20260514_turbopuffer.json) | 217.9892 | 204.8393 | [217.9892](raw_results/turbopuffer_unpinned/scalar_label_filter/10p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_10p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2660s / 0.3604s | 1.1414s / 1.1691s | 2.3697s / 2.4357s | 309,200 | measured |
| 20% | IDs only | [0.9315](raw_results/turbopuffer_unpinned/scalar_label_filter/20p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_20p_ids_only_serial_recall_20260514_turbopuffer.json) | 230.4684 | 224.7899 | [230.4684](raw_results/turbopuffer_unpinned/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_20p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2540s / 0.3439s | 1.1282s / 1.1460s | 2.3412s / 2.4115s | 2,000 | measured |
| 20% | scalar label | [0.9315](raw_results/turbopuffer_unpinned/scalar_label_filter/20p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_20p_scalar_label_serial_recall_20260514_turbopuffer.json) | 225.5087 | 209.9440 | [225.5087](raw_results/turbopuffer_unpinned/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_20p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2584s / 0.3722s | 1.1332s / 1.1609s | 2.3726s / 2.4220s | 3,600 | measured |
| 20% | vector | [0.9315](raw_results/turbopuffer_unpinned/scalar_label_filter/20p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_20p_vector_serial_recall_20260514_turbopuffer.json) | 201.4927 | 196.6834 | [201.4927](raw_results/turbopuffer_unpinned/scalar_label_filter/20p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_20p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2895s / 0.3860s | 1.1571s / 1.1837s | 2.4120s / 2.4709s | 309,200 | measured |
| 50% | IDs only | [0.9510](raw_results/turbopuffer_unpinned/scalar_label_filter/50p/ids_only/serial_recall/result_20260514_turbopuffer_scalar_label_filter_50p_ids_only_serial_recall_20260514_turbopuffer.json) | 186.2550 | 197.3191 | [197.3191](raw_results/turbopuffer_unpinned/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_50p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.3139s / 0.3770s | 1.1568s / 1.1653s | 2.4112s / 2.4309s | 2,000 | measured |
| 50% | scalar label | [0.9510](raw_results/turbopuffer_unpinned/scalar_label_filter/50p/scalar_label/serial_recall/result_20260514_turbopuffer_scalar_label_filter_50p_scalar_label_serial_recall_20260514_turbopuffer.json) | 212.0638 | 202.1825 | [212.0638](raw_results/turbopuffer_unpinned/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_50p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2779s / 0.3791s | 1.1395s / 1.1635s | 2.3942s / 2.4210s | 3,600 | measured |
| 50% | vector | [0.9510](raw_results/turbopuffer_unpinned/scalar_label_filter/50p/vector/serial_recall/result_20260514_turbopuffer_scalar_label_filter_50p_vector_serial_recall_20260514_turbopuffer.json) | 195.5719 | 191.3539 | [195.5719](raw_results/turbopuffer_unpinned/scalar_label_filter/50p/vector/concurrent_qps/result_20260514_turbopuffer_scalar_label_filter_50p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.2983s / 0.4003s | 1.1626s / 1.1994s | 2.4123s / 2.4736s | 309,200 | measured |

## Turbopuffer Pinned

These results are for the `laion100m_bulk` namespace pinned with 2 replicas. Namespace pinning can incur ongoing cost. Only change pinning state after an explicit instruction.

| Item | Value |
|---|---|
| Namespace | `laion100m_bulk` |
| Pinning | 2 replicas |
| Scalar label field | `label` |
| Logical row count | 100,000,000 |
| Concurrency | `60,80` |
| Concurrent duration | 30s per concurrency |
| Raw JSON count | 84 files |
| Result source | VDBBench `CloudPayloadSearchCase`, completed 2026-05-14 09:34:19 UTC |

### Unfiltered Search

| Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| IDs only | [0.9321](raw_results/turbopuffer_pinned/unfiltered/na/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_unfiltered_na_ids_only_serial_recall_20260514_turbopuffer.json) | 68.1714 | 67.3759 | [68.1714](raw_results/turbopuffer_pinned/unfiltered/na/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_unfiltered_na_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.8622s / 1.1541s | 2.1901s / 2.8368s | 2.4945s / 3.2979s | measured pinned 2 replicas |
| scalar label | [0.9321](raw_results/turbopuffer_pinned/unfiltered/na/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_unfiltered_na_scalar_label_serial_recall_20260514_turbopuffer.json) | 66.3255 | 63.0274 | [66.3255](raw_results/turbopuffer_pinned/unfiltered/na/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_unfiltered_na_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.8905s / 1.2454s | 1.7307s / 2.9217s | 2.0111s / 3.4858s | measured pinned 2 replicas |
| vector | [0.9321](raw_results/turbopuffer_pinned/unfiltered/na/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_unfiltered_na_vector_serial_recall_20260514_turbopuffer.json) | 71.6683 | 68.1989 | [71.6683](raw_results/turbopuffer_pinned/unfiltered/na/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_unfiltered_na_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.8222s / 1.1459s | 2.2470s / 2.6538s | 2.5866s / 3.0742s | measured pinned 2 replicas |

### Integer Filtered Search

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 99.9% | IDs only | [0.9436](raw_results/turbopuffer_pinned/int_filter/0_1p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_0_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 27.2995 | 27.2696 | [27.2995](raw_results/turbopuffer_pinned/int_filter/0_1p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_0_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1237s / 2.8024s | 4.6388s / 5.5416s | 5.1855s / 6.3376s | measured pinned 2 replicas |
| 99.9% | scalar label | [0.9436](raw_results/turbopuffer_pinned/int_filter/0_1p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_0_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 24.4802 | 24.6137 | [24.6137](raw_results/turbopuffer_pinned/int_filter/0_1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_0_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.3693s / 3.0890s | 4.7736s / 5.7601s | 5.3117s / 6.4807s | measured pinned 2 replicas |
| 99.9% | vector | [0.9436](raw_results/turbopuffer_pinned/int_filter/0_1p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_0_1p_vector_serial_recall_20260514_turbopuffer.json) | 26.7097 | 25.9503 | [26.7097](raw_results/turbopuffer_pinned/int_filter/0_1p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_0_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1616s / 2.9760s | 3.6554s / 4.8739s | 4.2209s / 5.1790s | measured pinned 2 replicas |
| 99% | IDs only | [0.7119](raw_results/turbopuffer_pinned/int_filter/1p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 57.9279 | 53.9051 | [57.9279](raw_results/turbopuffer_pinned/int_filter/1p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.0181s / 1.4454s | 2.3318s / 2.3129s | 2.5641s / 2.5376s | measured pinned 2 replicas |
| 99% | scalar label | [0.7119](raw_results/turbopuffer_pinned/int_filter/1p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 55.0892 | 55.5083 | [55.5083](raw_results/turbopuffer_pinned/int_filter/1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.0709s / 1.4044s | 2.5140s / 3.3094s | 2.8694s / 3.5411s | measured pinned 2 replicas |
| 99% | vector | [0.7119](raw_results/turbopuffer_pinned/int_filter/1p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_1p_vector_serial_recall_20260514_turbopuffer.json) | 54.9657 | 62.8649 | [62.8649](raw_results/turbopuffer_pinned/int_filter/1p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.0708s / 1.2418s | 2.6228s / 2.8078s | 2.8111s / 3.1029s | measured pinned 2 replicas |
| 90% | IDs only | [0.9076](raw_results/turbopuffer_pinned/int_filter/10p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_10p_ids_only_serial_recall_20260514_turbopuffer.json) | 41.9045 | 44.4073 | [44.4073](raw_results/turbopuffer_pinned/int_filter/10p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_10p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.3953s / 1.7418s | 3.2486s / 3.5434s | 3.6216s / 3.8602s | measured pinned 2 replicas |
| 90% | scalar label | [0.9076](raw_results/turbopuffer_pinned/int_filter/10p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_10p_scalar_label_serial_recall_20260514_turbopuffer.json) | 38.6995 | 38.8922 | [38.8922](raw_results/turbopuffer_pinned/int_filter/10p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_10p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.5157s / 2.0065s | 3.1624s / 3.6367s | 3.6246s / 3.9921s | measured pinned 2 replicas |
| 90% | vector | [0.9076](raw_results/turbopuffer_pinned/int_filter/10p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_10p_vector_serial_recall_20260514_turbopuffer.json) | 41.3297 | 43.5288 | [43.5288](raw_results/turbopuffer_pinned/int_filter/10p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_10p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.4182s / 1.7876s | 3.2278s / 4.1487s | 3.6074s / 4.5429s | measured pinned 2 replicas |
| 50% | IDs only | [0.9507](raw_results/turbopuffer_pinned/int_filter/50p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_50p_ids_only_serial_recall_20260514_turbopuffer.json) | 37.9533 | 36.9144 | [37.9533](raw_results/turbopuffer_pinned/int_filter/50p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_50p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.5442s / 2.1126s | 3.5208s / 4.3545s | 3.9063s / 4.8603s | measured pinned 2 replicas |
| 50% | scalar label | [0.9507](raw_results/turbopuffer_pinned/int_filter/50p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_50p_scalar_label_serial_recall_20260514_turbopuffer.json) | 34.4056 | 31.9928 | [34.4056](raw_results/turbopuffer_pinned/int_filter/50p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_50p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.7016s / 2.4362s | 3.9810s / 4.8653s | 4.3281s / 5.2590s | measured pinned 2 replicas |
| 50% | vector | [0.9507](raw_results/turbopuffer_pinned/int_filter/50p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_int_filter_50p_vector_serial_recall_20260514_turbopuffer.json) | 39.3695 | 35.5211 | [39.3695](raw_results/turbopuffer_pinned/int_filter/50p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_int_filter_50p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.4907s / 2.1941s | 3.6536s / 4.0588s | 4.0879s / 4.5410s | measured pinned 2 replicas |

### Scalar Label Filtered Search

| Filter rate | Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Payload bytes/query | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 0.1% | IDs only | [0.9438](raw_results/turbopuffer_pinned/scalar_label_filter/0_1p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 24.4910 | 26.4219 | [26.4219](raw_results/turbopuffer_pinned/scalar_label_filter/0_1p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.3628s / 2.8961s | 4.9708s / 6.1955s | 5.3354s / 6.6852s | 2,000 | measured pinned 2 replicas |
| 0.1% | scalar label | [0.9438](raw_results/turbopuffer_pinned/scalar_label_filter/0_1p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 24.0199 | 23.7316 | [24.0199](raw_results/turbopuffer_pinned/scalar_label_filter/0_1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.4117s / 3.2152s | 5.7264s / 7.0596s | 6.2560s / 7.6352s | 3,600 | measured pinned 2 replicas |
| 0.1% | vector | [0.9438](raw_results/turbopuffer_pinned/scalar_label_filter/0_1p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_1p_vector_serial_recall_20260514_turbopuffer.json) | 27.0399 | 24.2034 | [27.0399](raw_results/turbopuffer_pinned/scalar_label_filter/0_1p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1334s / 3.1734s | 4.6917s / 7.0679s | 5.1913s / 7.6104s | 309,200 | measured pinned 2 replicas |
| 0.2% | IDs only | [0.9375](raw_results/turbopuffer_pinned/scalar_label_filter/0_2p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_2p_ids_only_serial_recall_20260514_turbopuffer.json) | 10.8680 | 10.4027 | [10.8680](raw_results/turbopuffer_pinned/scalar_label_filter/0_2p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_2p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 5.1983s / 7.1515s | 8.2891s / 12.7924s | 8.7420s / 13.4412s | 2,000 | measured pinned 2 replicas |
| 0.2% | scalar label | [0.9375](raw_results/turbopuffer_pinned/scalar_label_filter/0_2p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_2p_scalar_label_serial_recall_20260514_turbopuffer.json) | 10.6353 | 10.7342 | [10.7342](raw_results/turbopuffer_pinned/scalar_label_filter/0_2p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_2p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 5.3313s / 7.1062s | 8.6504s / 9.8753s | 8.9867s / 10.2494s | 3,600 | measured pinned 2 replicas |
| 0.2% | vector | [0.9375](raw_results/turbopuffer_pinned/scalar_label_filter/0_2p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_2p_vector_serial_recall_20260514_turbopuffer.json) | 11.2150 | 10.7409 | [11.2150](raw_results/turbopuffer_pinned/scalar_label_filter/0_2p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_2p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 5.0762s / 6.9564s | 9.2954s / 9.5047s | 9.9218s / 9.9680s | 309,200 | measured pinned 2 replicas |
| 0.5% | IDs only | [0.5884](raw_results/turbopuffer_pinned/scalar_label_filter/0_5p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_5p_ids_only_serial_recall_20260514_turbopuffer.json) | 87.6429 | 86.1364 | [87.6429](raw_results/turbopuffer_pinned/scalar_label_filter/0_5p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_5p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.6726s / 0.9068s | 1.6049s / 2.1064s | 1.8145s / 2.3334s | 2,000 | measured pinned 2 replicas |
| 0.5% | scalar label | [0.5884](raw_results/turbopuffer_pinned/scalar_label_filter/0_5p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_5p_scalar_label_serial_recall_20260514_turbopuffer.json) | 76.2756 | 89.4607 | [89.4607](raw_results/turbopuffer_pinned/scalar_label_filter/0_5p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_5p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.7734s / 0.8754s | 1.6057s / 2.1874s | 1.8213s / 2.4023s | 3,600 | measured pinned 2 replicas |
| 0.5% | vector | [0.5884](raw_results/turbopuffer_pinned/scalar_label_filter/0_5p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_5p_vector_serial_recall_20260514_turbopuffer.json) | 90.7709 | 87.6651 | [90.7709](raw_results/turbopuffer_pinned/scalar_label_filter/0_5p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_0_5p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 0.6502s / 0.8896s | 1.6709s / 1.6307s | 1.8853s / 1.9052s | 309,200 | measured pinned 2 replicas |
| 1% | IDs only | [0.7081](raw_results/turbopuffer_pinned/scalar_label_filter/1p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_1p_ids_only_serial_recall_20260514_turbopuffer.json) | 52.4628 | 49.5000 | [52.4628](raw_results/turbopuffer_pinned/scalar_label_filter/1p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_1p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.1237s / 1.5734s | 2.6710s / 3.5404s | 2.9356s / 3.9195s | 2,000 | measured pinned 2 replicas |
| 1% | scalar label | [0.7081](raw_results/turbopuffer_pinned/scalar_label_filter/1p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_1p_scalar_label_serial_recall_20260514_turbopuffer.json) | 44.1460 | 44.5008 | [44.5008](raw_results/turbopuffer_pinned/scalar_label_filter/1p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_1p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.3339s / 1.7596s | 2.4121s / 3.4430s | 2.6916s / 3.7971s | 3,600 | measured pinned 2 replicas |
| 1% | vector | [0.7081](raw_results/turbopuffer_pinned/scalar_label_filter/1p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_1p_vector_serial_recall_20260514_turbopuffer.json) | 49.3199 | 50.3759 | [50.3759](raw_results/turbopuffer_pinned/scalar_label_filter/1p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_1p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.1911s / 1.5493s | 2.1767s / 3.5515s | 2.5939s / 3.8662s | 309,200 | measured pinned 2 replicas |
| 2% | IDs only | [0.7976](raw_results/turbopuffer_pinned/scalar_label_filter/2p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_2p_ids_only_serial_recall_20260514_turbopuffer.json) | 33.7544 | 31.0356 | [33.7544](raw_results/turbopuffer_pinned/scalar_label_filter/2p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_2p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.7305s / 2.4884s | 3.8347s / 5.3014s | 4.2140s / 5.7321s | 2,000 | measured pinned 2 replicas |
| 2% | scalar label | [0.7976](raw_results/turbopuffer_pinned/scalar_label_filter/2p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_2p_scalar_label_serial_recall_20260514_turbopuffer.json) | 29.1182 | 29.2327 | [29.2327](raw_results/turbopuffer_pinned/scalar_label_filter/2p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_2p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.0149s / 2.6609s | 3.9167s / 4.9900s | 4.3933s / 5.6679s | 3,600 | measured pinned 2 replicas |
| 2% | vector | [0.7976](raw_results/turbopuffer_pinned/scalar_label_filter/2p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_2p_vector_serial_recall_20260514_turbopuffer.json) | 30.3101 | 31.9835 | [31.9835](raw_results/turbopuffer_pinned/scalar_label_filter/2p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_2p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.9339s / 2.4337s | 3.6424s / 3.8990s | 4.0252s / 4.4262s | 309,200 | measured pinned 2 replicas |
| 5% | IDs only | [0.8746](raw_results/turbopuffer_pinned/scalar_label_filter/5p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_5p_ids_only_serial_recall_20260514_turbopuffer.json) | 27.0331 | 26.4031 | [27.0331](raw_results/turbopuffer_pinned/scalar_label_filter/5p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_5p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1529s / 2.9082s | 4.7154s / 4.8079s | 5.0599s / 5.2112s | 2,000 | measured pinned 2 replicas |
| 5% | scalar label | [0.8746](raw_results/turbopuffer_pinned/scalar_label_filter/5p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_5p_scalar_label_serial_recall_20260514_turbopuffer.json) | 27.4702 | 24.0167 | [27.4702](raw_results/turbopuffer_pinned/scalar_label_filter/5p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_5p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1413s / 3.2301s | 4.7670s / 6.8624s | 5.1783s / 7.4279s | 3,600 | measured pinned 2 replicas |
| 5% | vector | [0.8746](raw_results/turbopuffer_pinned/scalar_label_filter/5p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_5p_vector_serial_recall_20260514_turbopuffer.json) | 26.5261 | 27.9640 | [27.9640](raw_results/turbopuffer_pinned/scalar_label_filter/5p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_5p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.2085s / 2.7986s | 4.7486s / 4.7333s | 5.2843s / 5.1313s | 309,200 | measured pinned 2 replicas |
| 10% | IDs only | [0.9060](raw_results/turbopuffer_pinned/scalar_label_filter/10p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_10p_ids_only_serial_recall_20260514_turbopuffer.json) | 30.2780 | 28.3541 | [30.2780](raw_results/turbopuffer_pinned/scalar_label_filter/10p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_10p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.9348s / 2.7368s | 3.9769s / 4.3111s | 4.3750s / 4.7139s | 2,000 | measured pinned 2 replicas |
| 10% | scalar label | [0.9060](raw_results/turbopuffer_pinned/scalar_label_filter/10p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_10p_scalar_label_serial_recall_20260514_turbopuffer.json) | 27.0414 | 28.2824 | [28.2824](raw_results/turbopuffer_pinned/scalar_label_filter/10p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_10p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1782s / 2.7487s | 4.1424s / 5.2953s | 4.6148s / 5.9303s | 3,600 | measured pinned 2 replicas |
| 10% | vector | [0.9060](raw_results/turbopuffer_pinned/scalar_label_filter/10p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_10p_vector_serial_recall_20260514_turbopuffer.json) | 27.4952 | 26.1624 | [27.4952](raw_results/turbopuffer_pinned/scalar_label_filter/10p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_10p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.1285s / 2.9510s | 4.7865s / 5.1223s | 5.3199s / 5.4753s | 309,200 | measured pinned 2 replicas |
| 20% | IDs only | [0.9315](raw_results/turbopuffer_pinned/scalar_label_filter/20p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_20p_ids_only_serial_recall_20260514_turbopuffer.json) | 30.4973 | 30.7461 | [30.7461](raw_results/turbopuffer_pinned/scalar_label_filter/20p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_20p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.9175s / 2.5179s | 4.4305s / 4.7006s | 4.8306s / 5.1351s | 2,000 | measured pinned 2 replicas |
| 20% | scalar label | [0.9315](raw_results/turbopuffer_pinned/scalar_label_filter/20p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_20p_scalar_label_serial_recall_20260514_turbopuffer.json) | 28.9875 | 30.1028 | [30.1028](raw_results/turbopuffer_pinned/scalar_label_filter/20p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_20p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 2.0397s / 2.5825s | 3.9483s / 5.1102s | 4.3334s / 5.6086s | 3,600 | measured pinned 2 replicas |
| 20% | vector | [0.9315](raw_results/turbopuffer_pinned/scalar_label_filter/20p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_20p_vector_serial_recall_20260514_turbopuffer.json) | 29.2896 | 30.0731 | [30.0731](raw_results/turbopuffer_pinned/scalar_label_filter/20p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_20p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.9961s / 2.5901s | 3.4770s / 4.4540s | 3.9894s / 4.8944s | 309,200 | measured pinned 2 replicas |
| 50% | IDs only | [0.9510](raw_results/turbopuffer_pinned/scalar_label_filter/50p/ids_only/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_50p_ids_only_serial_recall_20260514_turbopuffer.json) | 31.8482 | 31.2622 | [31.8482](raw_results/turbopuffer_pinned/scalar_label_filter/50p/ids_only/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_50p_ids_only_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.8387s / 2.4725s | 4.0837s / 4.4324s | 4.4679s / 5.0357s | 2,000 | measured pinned 2 replicas |
| 50% | scalar label | [0.9510](raw_results/turbopuffer_pinned/scalar_label_filter/50p/scalar_label/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_50p_scalar_label_serial_recall_20260514_turbopuffer.json) | 29.9535 | 30.7861 | [30.7861](raw_results/turbopuffer_pinned/scalar_label_filter/50p/scalar_label/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_50p_scalar_label_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.9584s / 2.5332s | 4.0164s / 5.2395s | 4.4023s / 5.8014s | 3,600 | measured pinned 2 replicas |
| 50% | vector | [0.9510](raw_results/turbopuffer_pinned/scalar_label_filter/50p/vector/serial_recall/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_50p_vector_serial_recall_20260514_turbopuffer.json) | 32.4054 | 29.9090 | [32.4054](raw_results/turbopuffer_pinned/scalar_label_filter/50p/vector/concurrent_qps/result_20260514_turbopuffer_pinned_2rep_scalar_label_filter_50p_vector_topk100_c60_c80_30s_20260514_turbopuffer.json) | 1.8102s / 2.5986s | 4.1656s / 5.5147s | 4.5870s / 5.9656s | 309,200 | measured pinned 2 replicas |

## Run Queue

Before running more jobs, select the next subset explicitly. Suggested axes:

| Axis | Options |
|---|---|
| Product | Tiered 4CU, Capacity 12CU, Pinecone serverless, Turbopuffer unpinned, Turbopuffer pinned |
| Filter mode | unfiltered, integer filtered, scalar label filtered |
| Filter rate | one or more rates from the planned filter-rate table |
| Payload | IDs only, scalar label, vector |
| Concurrency | default `60,80` unless changed |
| Duration | default 60s unless changed |
| Required order | serial recall first, concurrent QPS second |

The current pause point is after Tiered 4CU scalar label filter at 1% for all
three payload profiles. Those three rows have concurrent throughput results,
but still need serial recall runs before they count as complete matrix rows.
