# FTS BM25 End-to-End Testing Plan

This plan validates the first-pass BM25 full-text-search implementation across
the supported backend and dataset matrix.

## Scope

Backends:

- Milvus
- ElasticCloud
- Vespa
- TurboPuffer

Default datasets:

- MS MARCO Small (100K documents)
- MS MARCO Medium (1M documents)
- HotpotQA Small (100K documents)
- HotpotQA Medium (1M documents)

Out of scope for the first E2E pass:

- Large datasets:
  - MS MARCO Large (8.8M documents)
  - HotpotQA Large (5.2M documents)
- Payload/text retrieval.
- ClickHouse token/boolean FTS.
- Hybrid dense+sparse retrieval.

## Service Availability Notes

Elastic:

- Elastic Cloud Hosted is a paid managed service with a free 14-day trial. The
  official pricing page describes hosted pricing as resource based and
  pay-as-you-go.
- Elastic self-managed has a "Free and open" tier. This is a viable no-service
  option for Elasticsearch itself, but the current VDBBench command in this
  branch is `elasticcloudhnsw`, which takes Elastic Cloud credentials
  (`--cloud-id`, `--password`) rather than a self-managed host URL.
- For this E2E plan, Elastic uses ElasticCloud credentials. If we want a
  no-charge self-managed Elastic run, we should first add or verify host/port
  CLI support for this backend.

References:

- https://www.elastic.co/pricing
- https://www.elastic.co/pricing/faq
- https://www.elastic.co/pricing/self-managed
- https://www.elastic.co/elasticsearch/service

Vespa:

- Vespa is available as open source and can be self-hosted.
- Vespa Cloud is a managed service with pricing by allocated resources. The
  free trial includes usage credits and stops the application when credits run
  out rather than billing beyond the credits.
- The current VDBBench `vespa` command takes `--uri` and `--port`, so the
  recommended first E2E target is a self-hosted/local Vespa endpoint.

References:

- https://blog.vespa.ai/open-sourcing-vespa-yahoos-big-data-processing/
- https://vespa.ai/free-trial/
- https://cloud.vespa.ai/price-calculator
- https://cloud.vespa.ai/

## Common Environment

Set these before running the matrix:

```bash
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset

export MILVUS_URI="http://..."
export MILVUS_USER=""
export MILVUS_PASSWORD=""

export ELASTIC_CLOUD_ID="..."
export ELASTIC_PASSWORD="..."

export VESPA_URI="http://127.0.0.1"
export VESPA_PORT="8080"

export TURBOPUFFER_API_KEY="..."
export TURBOPUFFER_API_BASE_URL="https://api.turbopuffer.com"
export TURBOPUFFER_NAMESPACE_PREFIX="vdbbench-fts-e2e"
```

Use `python3.11 -m vectordb_bench.cli.vectordbbench` from the repository root.
Do not use bare `pytest` or an unpinned Python executable for local validation.

## Common Benchmark Options

Each E2E run should include:

```bash
--case-type FTSmsmarcoPerformance
--drop-old
--load
--search-serial
--search-concurrent
--k 100
--concurrency-duration 30
--num-concurrency "1,5,10,20"
--concurrency-timeout 3600
```

The default VDBBench concurrency list is larger (`1,5,10,20,30,40,60,80`).
For the first functional E2E pass, `1,5,10,20` is enough to exercise concurrent
search without turning validation into a full leaderboard run.

## Preflight

Run the focused FTS unit/integration suite:

```bash
python3.11 -m pytest \
  tests/test_fts_dataset.py \
  tests/test_fts_cases.py \
  tests/test_fts_runners.py \
  tests/test_fts_backend_capability.py \
  tests/test_fts_milvus.py \
  tests/test_fts_elastic_cloud.py \
  tests/test_fts_vespa.py \
  tests/test_fts_turbopuffer.py \
  tests/test_fts_format_results.py \
  -q
```

Expected result:

- All focused FTS tests pass.
- Current known local warning from `pytz` is acceptable.

Run one dry-run per backend to verify CLI argument parsing and case config
selection:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --dry-run \
  --uri "$MILVUS_URI" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --dry-run \
  --cloud-id "$ELASTIC_CLOUD_ID" \
  --password "$ELASTIC_PASSWORD" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --dry-run \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --dry-run \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-dryrun" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"
```

Note: `elasticcloudhnsw`, `vespa`, and `turbopuffer` are existing backend CLI
commands. When `--case-type FTSmsmarcoPerformance` is selected, the shared CLI
logic swaps their DB case config to the backend FTS config.

## Milvus Matrix

```bash
python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## ElasticCloud Matrix

```bash
python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --cloud-id "$ELASTIC_CLOUD_ID" \
  --password "$ELASTIC_PASSWORD" \
  --task-label "fts-e2e-elastic-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --cloud-id "$ELASTIC_CLOUD_ID" \
  --password "$ELASTIC_PASSWORD" \
  --task-label "fts-e2e-elastic-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --cloud-id "$ELASTIC_CLOUD_ID" \
  --password "$ELASTIC_PASSWORD" \
  --task-label "fts-e2e-elastic-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --cloud-id "$ELASTIC_CLOUD_ID" \
  --password "$ELASTIC_PASSWORD" \
  --task-label "fts-e2e-elastic-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## Vespa Matrix

```bash
python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## TurboPuffer Matrix

Use a separate namespace for each run so cleanup is isolated:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-msmarco-small" \
  --task-label "fts-e2e-tpuf-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-msmarco-medium" \
  --task-label "fts-e2e-tpuf-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-hotpotqa-small" \
  --task-label "fts-e2e-tpuf-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-hotpotqa-medium" \
  --task-label "fts-e2e-tpuf-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## Validation After Each Run

For each task label:

1. Confirm the command exits with status `0`.
2. Confirm a result JSON is written under `vectordb_bench/results/<DB>/`.
3. Confirm the result JSON is readable by `TestResult.read_file`.
4. Confirm the performance metrics include:
   - `insert_duration`
   - `optimize_duration`
   - `load_duration`
   - `recall`
   - `ndcg`
   - `mrr`
   - `serial_latency_p99`
   - `serial_latency_p95`
   - `qps`
   - `conc_qps_list`
   - `conc_latency_p99_list`
   - `conc_latency_p95_list`
5. Confirm logs do not contain:
   - `NotImplementedError`
   - wrong DB case config type errors
   - dataset/qrel validation errors
   - systemic empty search result failures
   - insert retry exhaustion

## Stop Conditions

Stop the matrix and debug before continuing if any backend has:

- insert failure after retries
- optimize timeout
- serial search failure
- all-zero `recall`, `ndcg`, and `mrr` for a completed run
- missing or unreadable result JSON
- backend API/schema error indicating adapter incompatibility

## Optional Phase: Large Datasets

Run only after all default dataset runs pass:

- MS MARCO Large (8.8M documents)
- HotpotQA Large (5.2M documents)

Use the same backend commands and replace `--dataset-with-size-type` with the
Large tier. Treat this as capacity/performance validation, not first-pass
functional validation.
