# Cloud Insert Results

This folder records VDBBench `CloudInsertCase` results for the cloud leaderboard
insert-only test line. Raw JSON artifacts are copied exactly from the local
VectorDBBench result output under `cloud_insert/raw_results/`.

## Framework

| Field | Value |
|---|---|
| Framework repo | `https://github.com/jamesgao-jpg/VectorDBBench` |
| Branch | `cloud-payload-search-case` |
| Commit | `f61b4471275dc5241b36b0e5ba765494726dd613` |
| State | Dirty local workspace; source result JSONs were untracked in the local checkout. |

## Case

| Field | Value |
|---|---|
| Case type | `CloudInsertCase` |
| Case id | `600` |
| Dataset | `Medium Cohere (768dim, 1M)` |
| Insert rows | `1,000,000` |
| Batch size | `1,000` |
| Stages | `drop_old`, `load` |

## Results

| Product | Mode | Date UTC | Load concurrency | Rows/s | Completion seconds | Indexed-after-searchable seconds | Raw JSON |
|---|---:|---|---:|---:|---:|---:|---|
| Zilliz Cloud tiered 4CU | AUTOINDEX | 2026-05-12 | 8 | 6850.5883 | 145.9729 | 50.6191 | `cloud_insert/raw_results/zilliz_cloud_tiered_4cu/result_20260512_cloud_insert_zilliz_tiered_cohere_1m_bs1k_conc8_20260512_zillizcloud.json` |
| Zilliz Cloud tiered 4CU | AUTOINDEX | 2026-05-11 | 0 | 3919.9296 | 255.1066 | 28.5956 | `cloud_insert/raw_results/zilliz_cloud_tiered_4cu/result_20260511_cloud_insert_zilliz_tiered_cohere_1m_bs1k_20260511_zillizcloud.json` |
| TurboPuffer | backpressure disabled | 2026-05-11 | 0 | 519.3670 | 1925.4209 | 187.3016 | `cloud_insert/raw_results/turbopuffer/result_20260511_cloud_insert_turbopuffer_bp_off_cohere_1m_bs1k_20260511_turbopuffer.json` |
| TurboPuffer | backpressure enabled | 2026-05-11 | 0 | 485.4639 | 2059.8853 | 182.2381 | `cloud_insert/raw_results/turbopuffer/result_20260511_cloud_insert_turbopuffer_bp_on_cohere_1m_bs1k_20260511_turbopuffer.json` |
| Pinecone Serverless | serverless | 2026-05-12 | 8 | 426.9220 | 2342.3481 | 0.0001 | `cloud_insert/raw_results/pinecone_serverless/result_20260512_cloud_insert_pinecone_cohere_1m_bs1k_conc8_20260512_pinecone.json` |
| Pinecone Serverless | serverless | 2026-05-11 | 0 | 222.1111 | 4502.2520 | 0.0001 | `cloud_insert/raw_results/pinecone_serverless/result_20260511_cloud_insert_pinecone_cohere_1m_bs1k_20260511_pinecone.json` |

All six runs finished with VDBBench result label `:)`.

## Manifest

`cloud_insert/manifest.jsonl` contains one JSON object per raw result file. The
manifest includes case metadata, run identifiers, raw file paths, insert metrics,
and the framework branch/commit used for the source run.

## Reproduction Commands

These commands redact credentials and private endpoints with placeholders. Keep
the same task labels and DB labels when reproducing the archived runs.

```bash
.venv/bin/python -m vectordb_bench.cli.vectordbbench zillizautoindex \
  --uri '<uri>' \
  --user-name db_admin \
  --password '<password>' \
  --token '<token>' \
  --collection-name cloud_insert_cohere_1m_tiered_20260511 \
  --db-label zilliz_cloud_tiered_cloud_insert_cohere_1m_bs1k \
  --task-label cloud_insert_zilliz_tiered_cohere_1m_bs1k_20260511 \
  --case-type CloudInsertCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --cloud-insert-batch-size 1000 \
  --load-concurrency 0 \
  --skip-search-serial \
  --skip-search-concurrent

.venv/bin/python -m vectordb_bench.cli.vectordbbench zillizautoindex \
  --uri '<uri>' \
  --user-name db_admin \
  --password '<password>' \
  --token '<token>' \
  --collection-name cloud_insert_cohere_1m_tiered_conc8_20260512 \
  --db-label zilliz_cloud_tiered_cloud_insert_cohere_1m_bs1k_conc8 \
  --task-label cloud_insert_zilliz_tiered_cohere_1m_bs1k_conc8_20260512 \
  --case-type CloudInsertCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --cloud-insert-batch-size 1000 \
  --load-concurrency 8 \
  --skip-search-serial \
  --skip-search-concurrent

.venv/bin/python -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key '<api-key>' \
  --region aws-us-west-2 \
  --namespace cloud_insert_cohere_1m_bp_off_20260511 \
  --disable-backpressure \
  --db-label turbopuffer_bp_off_cloud_insert_cohere_1m_bs1k \
  --task-label cloud_insert_turbopuffer_bp_off_cohere_1m_bs1k_20260511 \
  --case-type CloudInsertCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --cloud-insert-batch-size 1000 \
  --load-concurrency 0 \
  --skip-search-serial \
  --skip-search-concurrent

.venv/bin/python -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key '<api-key>' \
  --region aws-us-west-2 \
  --namespace cloud_insert_cohere_1m_bp_on_20260511 \
  --enable-backpressure \
  --db-label turbopuffer_bp_on_cloud_insert_cohere_1m_bs1k \
  --task-label cloud_insert_turbopuffer_bp_on_cohere_1m_bs1k_20260511 \
  --case-type CloudInsertCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --cloud-insert-batch-size 1000 \
  --load-concurrency 0 \
  --skip-search-serial \
  --skip-search-concurrent

.venv/bin/python -m vectordb_bench.cli.vectordbbench pinecone \
  --api-key '<api-key>' \
  --index-name vdbbench-cloud-insert-cohere-1m \
  --db-label pinecone_cloud_insert_cohere_1m_bs1k \
  --task-label cloud_insert_pinecone_cohere_1m_bs1k_20260511 \
  --case-type CloudInsertCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --cloud-insert-batch-size 1000 \
  --load-concurrency 0 \
  --skip-search-serial \
  --skip-search-concurrent

.venv/bin/python -m vectordb_bench.cli.vectordbbench pinecone \
  --api-key '<api-key>' \
  --index-name vdbbench-cloud-insert-cohere-1m \
  --db-label pinecone_cloud_insert_cohere_1m_bs1k_conc8 \
  --task-label cloud_insert_pinecone_cohere_1m_bs1k_conc8_20260512 \
  --case-type CloudInsertCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --cloud-insert-batch-size 1000 \
  --load-concurrency 8 \
  --skip-search-serial \
  --skip-search-concurrent
```
