# Vespa HotpotQA FTS E2E Report

## Prequsites

- Backend: Vespa single-node container.
- Dataset family: HotpotQA.
- Current committed raw results: `HotpotQA Medium (1M documents)` and `HotpotQA Large (5.2M documents)` on the `r7i.4xlarge` server.
- Run dates represented here: 2026-06-02, 2026-06-03, and 2026-06-04.
- Source runbook: `docs/fts-backends/vespa.md`.
- Raw result directory: `raw_results/`.
- The current FTS CLI uses `FTSmsmarcoPerformance` as the generic FTS case type; the dataset is selected by `--dataset-with-size-type`.

### Physical Machine Stats

Client machine:

- EC2 type: `i8g.2xlarge`.
- OS: Ubuntu 22.04, Linux `6.8.0-1053-aws`, `aarch64`.
- CPU: 8 vCPU, Neoverse-V2, 1 thread per core.
- Memory: about 61 GiB RAM, no swap.
- Disk quota: `/dev/root` ext4, 485 GiB total, 102 GiB available at last check.
- Role: runs VectorDBBench from `/home/ubuntu/VectorDBBench`.

Server machine:

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Vespa container and `/srv/vespa` state directories.

## Server Setup

Validated deployment:

- Vespa image: `vespaengine/vespa:8.694.53`.
- Container name: `vespa`.
- Hostname: `vespa-container`.
- Persistent host directories: `/srv/vespa/var` and `/srv/vespa/logs`.
- Ports: `8080` for feed/query and `19071` for config/deploy.

### FTS Index, Analyzer, And Ranking Configuration

Vespa did not use an unconfigured product-default text schema. VDBBench explicitly deployed a Vespa application schema for FTS and explicitly queried it with a BM25 rank profile.

Effective schema:

- schema: `VectorDBBenchCollection`.
- `id`: `string`, indexed as `summary` and `attribute`.
- `text`: `string`, indexed as `index` and `summary`.
- `text` index setting: `enable-bm25`.
- rank profile: `bm25`, inheriting `default`, with first phase `bm25(text)`.

Effective query settings:

- query expression: `select id from VectorDBBenchCollection where userQuery()` for ids-only runs.
- query expression: `select id, text from VectorDBBenchCollection where userQuery()` for text-payload runs.
- `ranking=bm25`.
- `default-index=text`.
- `type=any`.
- `hits=k`.

Inherited Vespa product defaults where VDBBench did not override:

- BM25 rank feature parameters: `k1=1.2`, `b=0.75`.
- string index text processing, including tokenized text matching and normalization.
- stemming default: `best`.

`VespaFtsConfig` has no extra tunable index or search parameters, so Vespa raw results record `db_case_config={}`. Payload profile only changes returned fields; it does not change the Vespa index or rank profile. The relevant code paths are the Vespa application package/schema construction and query construction in `vectordb_bench/backend/clients/vespa/vespa.py`, plus empty `VespaFtsConfig` in `vectordb_bench/backend/clients/vespa/config.py`.

Reproducible fresh-deploy script:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo mkdir -p /srv/vespa/var /srv/vespa/logs
sudo chown -R 1000:1000 /srv/vespa/var /srv/vespa/logs

sudo docker rm -f vespa >/dev/null 2>&1 || true
sudo find /srv/vespa/var -mindepth 1 -delete
sudo find /srv/vespa/logs -mindepth 1 -delete
sudo docker pull vespaengine/vespa:8.694.53

sudo docker run -d --name vespa --user vespa:vespa --hostname vespa-container \
  --ulimit nofile=262144:262144 --pids-limit=-1 \
  -v /srv/vespa/var:/opt/vespa/var \
  -v /srv/vespa/logs:/opt/vespa/logs \
  -p 0.0.0.0:8080:8080 \
  -p 0.0.0.0:19071:19071 \
  --restart unless-stopped \
  vespaengine/vespa:8.694.53

curl -fsS http://127.0.0.1:19071/state/v1/health
```

Fresh teardown script used after the run:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f vespa >/dev/null 2>&1 || true
sudo find /srv/vespa/var -mindepth 1 -delete
sudo find /srv/vespa/logs -mindepth 1 -delete
sudo docker ps -a
sudo docker volume ls
```

## VDBBench Running

Exact client script for the committed HotpotQA runs:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "http://${SERVER_HOST}" \
  --port "8080" \
  --task-label "fts-e2e-vespa-hotpotqa-medium-r7i" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The committed HotpotQA Large run used the same command with task label `fts-e2e-vespa-hotpotqa-large-r7i` and dataset size `HotpotQA Large (5.2M documents)`.

Exact client script for the 2026-06-04 HotpotQA Medium ids-only and text-payload matrix:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_TAG="20260604T074646Z"

for PAYLOAD_PROFILE in ids_only text; do
  if [[ "${PAYLOAD_PROFILE}" == "ids_only" ]]; then
    LABEL_PAYLOAD="ids"
    PAYLOAD_ARGS=()
  else
    LABEL_PAYLOAD="text"
    PAYLOAD_ARGS=(--payload-profile text)
  fi

  python3.11 -m vectordb_bench.cli.vectordbbench vespa \
    --uri "http://${SERVER_HOST}" \
    --port "8080" \
    --task-label "fts-hotpotqa-medium-vespa-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "HotpotQA Medium (1M documents)" \
    "${PAYLOAD_ARGS[@]}" \
    --drop-old --load --search-serial --search-concurrent \
    --k 100 \
    --concurrency-duration 30 \
    --num-concurrency "1,10,20,40,60,80" \
    --concurrency-timeout 3600
done
```

Effective Vespa FTS case config from the raw JSON: no backend-specific case fields are set. The VDBBench Vespa adapter deploys the application package through port `19071` and queries through port `8080`.

## Result

Historical note: the text-payload matrix run `fts-matrix-vespa-hotpotqa-medium-text-c20-40-80-r7i-20260603T061706Z` is intentionally excluded from the result table because VDBBench emitted only a zero-metric failure placeholder JSON. Log evidence shows the run loaded successfully (`load_duration=580.4292s`) and completed concurrency 20 (`72.8836 QPS`), then hung during concurrency 40 with repeated Vespa `Summary data is incomplete` timeout warnings and was terminated with `RUN_FAILED_143`. The 2026-06-04 full six-concurrency text-payload rerun below completed successfully, though Vespa still emitted timeout warnings during concurrency 60 and 80.

| Raw JSON | Task label | Dataset size | Payload | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `result_20260602_fts-e2e-vespa-hotpotqa-medium-r7i_vespa.json` | `fts-e2e-vespa-hotpotqa-medium-r7i` | 1M | ids_only | 575.5304 | 80.0872 | 0.8309 | 0.7208 | 0.8500 | 0.2647 | 0.3261 | 1/5/10/20 | 6.4907 / 33.8533 / 57.9060 / 80.0872 |
| `result_20260604_fts-hotpotqa-medium-vespa-ids-c1-10-20-40-60-80-r7i-20260604T074646Z_vespa.json` | `fts-hotpotqa-medium-vespa-ids-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | ids_only | 579.2518 | 181.5240 | 0.8309 | 0.7208 | 0.8500 | 0.2628 | 0.3223 | 1/10/20/40/60/80 | 6.6652 / 58.1128 / 81.3796 / 104.5544 / 140.9124 / 181.5240 |
| `result_20260604_fts-hotpotqa-medium-vespa-text-c1-10-20-40-60-80-r7i-20260604T074646Z_vespa.json` | `fts-hotpotqa-medium-vespa-text-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | text | 579.3951 | 177.2947 | 0.8309 | 0.7208 | 0.8500 | 0.2683 | 0.3313 | 1/10/20/40/60/80 | 5.2029 / 55.8888 / 79.5598 / 100.9787 / 138.3337 / 177.2947 |
| `result_20260602_fts-e2e-vespa-hotpotqa-large-r7i_vespa.json` | `fts-e2e-vespa-hotpotqa-large-r7i` | 5.2M | ids_only | 2954.2589 | 46.3472 | 0.6754 | 0.5460 | 0.6640 | 0.4460 | 0.4465 | 1/5/10/20 | 3.3531 / 13.1559 / 24.4787 / 46.3472 |
