# Vespa MS MARCO FTS E2E Report

## Prequsites

- Backend: Vespa single-node container.
- Dataset family: MS MARCO.
- Current committed raw results: `MS MARCO Small (100K documents)` and `MS MARCO Medium (1M documents)` on the original `m5d.2xlarge` server and the later `r7i.4xlarge` server.
- Run dates represented here: 2026-05-28, 2026-06-01, 2026-06-02, 2026-06-03, and 2026-06-04.
- Source runbook: `docs/fts-backends/vespa.md`.
- Raw result directory: `raw_results/`.
- Current result JSONs have connection fields masked by VectorDBBench.

### Physical Machine Stats

Client machine:

- EC2 type: `i8g.2xlarge`.
- OS: Ubuntu 22.04, Linux `6.8.0-1053-aws`, `aarch64`.
- CPU: 8 vCPU, Neoverse-V2, 1 thread per core.
- Memory: about 61 GiB RAM, no swap.
- Disk quota: `/dev/root` ext4, 485 GiB total, 102 GiB available at last check.
- Role: runs VectorDBBench from `/home/ubuntu/VectorDBBench`.

Original server machine:

- EC2 type: `m5d.2xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 8 vCPU, Intel Xeon Platinum 8175M, 4 physical cores, 2 threads per core.
- Memory: about 30 GiB usable RAM plus 31 GiB swap.
- Disk quota: root XFS filesystem 500 GiB total with 451 GiB available at last check; additional 279 GiB NVMe mounted at `/data`.
- Docker: overlay2 under `/var/lib/docker` on the root filesystem.
- Role: runs the Vespa container and `/srv/vespa` state directories.

Rerun server machine:

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Vespa container and `/srv/vespa` state directories for the `r7i` rerun.

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

Fresh teardown script used after the `r7i` rerun:

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

Exact client script for the committed MS MARCO Small runs:

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
  --task-label "fts-e2e-vespa-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The `r7i.4xlarge` rerun used the same command with task label `fts-e2e-vespa-msmarco-small-r7i`.
The `r7i.4xlarge` medium run used the same command with task label `fts-e2e-vespa-msmarco-medium-r7i` and dataset size `MS MARCO Medium (1M documents)`.

Exact client script for the `r7i.4xlarge` MS MARCO Small text-payload run:

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
  --port 8080 \
  --task-label fts-e2e-vespa-msmarco-small-text-r7i \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --payload-profile text \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,10,20,40,60,80" \
  --concurrency-timeout 3600
```

Exact client script for the `r7i.4xlarge` MS MARCO Medium ids-only and text-payload matrix:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_TAG="20260604T041648Z"

for PAYLOAD_PROFILE in ids_only text; do
  if [[ "${PAYLOAD_PROFILE}" == "ids_only" ]]; then
    LABEL_PAYLOAD="ids"
  else
    LABEL_PAYLOAD="text"
  fi

  python3.11 -m vectordb_bench.cli.vectordbbench vespa \
    --uri "http://${SERVER_HOST}" \
    --port 8080 \
    --task-label "fts-msmarco-medium-vespa-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "MS MARCO Medium (1M documents)" \
    --payload-profile "${PAYLOAD_PROFILE}" \
    --drop-old --load --search-serial --search-concurrent \
    --k 100 --concurrency-duration 30 \
    --num-concurrency "1,10,20,40,60,80" \
    --concurrency-timeout 3600
done
```

Effective Vespa FTS case config from the raw JSON: no backend-specific case fields are set. The VDBBench Vespa adapter deploys the application package through port `19071` and queries through port `8080`.

## Result

| Raw JSON | Task label | Dataset size | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/5/10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260528_fts-e2e-vespa-msmarco-small_vespa.json` | `fts-e2e-vespa-msmarco-small` | 100K | 171.9693 | 478.5384 | 0.9416 | 0.7509 | 0.7015 | 0.0212 | 0.0269 | 63.4393 / 146.3972 / 428.5583 / 478.5384 |
| `result_20260601_fts-e2e-vespa-msmarco-small_vespa.json` | `fts-e2e-vespa-msmarco-small` | 100K | 81.6402 | 482.1899 | 0.9416 | 0.7509 | 0.7015 | 0.0202 | 0.0260 | 78.8898 / 336.8178 / 482.1899 / 470.0384 |
| `result_20260602_fts-e2e-vespa-msmarco-small-r7i_vespa.json` | `fts-e2e-vespa-msmarco-small-r7i` | 100K | 79.2473 | 734.5241 | 0.9416 | 0.7509 | 0.7015 | 0.0184 | 0.0230 | 91.7244 / 512.5352 / 347.9730 / 734.5241 |
| `result_20260602_fts-e2e-vespa-msmarco-medium-r7i_vespa.json` | `fts-e2e-vespa-msmarco-medium-r7i` | 1M | 585.7496 | 196.8213 | 0.8409 | 0.5499 | 0.4767 | 0.1268 | 0.1744 | 17.3137 / 85.2530 / 139.2205 / 196.8213 |

Latest `r7i.4xlarge` rerun vs previous `m5d.2xlarge` run:

- QPS increased from `482.1899` to `734.5241` (+52.3%).
- Load duration changed from `81.6402s` to `79.2473s` (-2.9%).
- Recall stayed unchanged at `0.9416`.
- p95 changed from `0.0202s` to `0.0184s`; p99 changed from `0.0260s` to `0.0230s`.

MS MARCO Small text payload rerun on `r7i.4xlarge`:

| Raw JSON | Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260603_fts-e2e-vespa-msmarco-small-text-r7i_vespa.json` | `fts-e2e-vespa-msmarco-small-text-r7i` | 78.8999 | 788.0555 | 0.9416 | 0.7509 | 0.7015 | 0.0193 | 0.0236 | 64.6508 / 786.3064 / 131.0499 / 788.0555 / 422.9538 / 365.3142 |

Text payload details:

- `payload_profile=text`.
- Returned fields: `id` and `text`.
- Estimated payload bytes per query from VectorDBBench: `53200`.
- The concurrency 20, 60, and 80 stages produced request-send warnings and lower QPS, so the peak came from concurrency 40.

MS MARCO Medium six-concurrency rerun on `r7i.4xlarge`:

| Raw JSON | Payload | Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260604_fts-msmarco-medium-vespa-ids-c1-10-20-40-60-80-r7i-20260604T041648Z_vespa.json` | `ids_only` | `fts-msmarco-medium-vespa-ids-c1-10-20-40-60-80-r7i-20260604T041648Z` | 581.5774 | 257.0647 | 0.8409 | 0.5499 | 0.4767 | 0.1231 | 0.1688 | 17.2619 / 153.1753 / 217.6157 / 230.2074 / 238.9556 / 257.0647 |
| `result_20260604_fts-msmarco-medium-vespa-text-c1-10-20-40-60-80-r7i-20260604T041648Z_vespa.json` | `text` | `fts-msmarco-medium-vespa-text-c1-10-20-40-60-80-r7i-20260604T041648Z` | 581.4244 | 251.4636 | 0.8409 | 0.5499 | 0.4767 | 0.1248 | 0.1702 | 15.0937 / 133.4407 / 199.3716 / 209.5499 / 234.2022 / 251.4636 |

MS MARCO Medium stability comparison against the previous `r7i.4xlarge` ids-only run:

- At the shared concurrency level `20`, QPS changed from `196.8213` to `217.6157` (+10.6%).
- The new six-concurrency run reached a higher peak QPS of `257.0647` at concurrency `80` (+30.6% vs the previous peak at concurrency `20`).
- Load duration changed from `585.7496s` to `581.5774s` (-0.7%).
- Recall stayed unchanged at `0.8409`.
- Text payload QPS was `251.4636`, which is 2.2% below the new ids-only rerun at the same concurrency list.
