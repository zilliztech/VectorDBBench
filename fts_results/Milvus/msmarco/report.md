# Milvus MS MARCO FTS E2E Report

## Prequsites

- Backend: Milvus standalone.
- Dataset family: MS MARCO.
- Current committed raw results: `MS MARCO Small (100K documents)` and `MS MARCO Medium (1M documents)` on the original `m5d.2xlarge` server and the later `r7i.4xlarge` server.
- Run dates represented here: 2026-05-28, 2026-06-01, 2026-06-02, 2026-06-03, and 2026-06-04.
- Source runbook: `docs/fts-backends/milvus.md`.
- Raw result directory: `raw_results/`.
- Current result JSONs have connection fields masked by VectorDBBench.

### Physical Machine Stats

Client machine:

- EC2 type: `i8g.2xlarge`.
- OS: Ubuntu 22.04, Linux `6.8.0-1053-aws`, `aarch64`.
- CPU: 8 vCPU, Neoverse-V2, 1 thread per core.
- Memory: about 61 GiB RAM, no swap.
- Disk quota: `/dev/root` ext4, 485 GiB total, 102 GiB available at last check.
- Role: runs VectorDBBench from `/home/ubuntu/VectorDBBench` and stores dataset cache under `/tmp/vectordb_bench/dataset`.

Original server machine:

- EC2 type: `m5d.2xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 8 vCPU, Intel Xeon Platinum 8175M, 4 physical cores, 2 threads per core.
- Memory: about 30 GiB usable RAM plus 31 GiB swap.
- Disk quota: root XFS filesystem 500 GiB total with 451 GiB available at last check; additional 279 GiB NVMe mounted at `/data`.
- Docker: overlay2 under `/var/lib/docker` on the root filesystem.
- Role: runs the Milvus standalone server deployment.

Rerun server machine:

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Milvus standalone server deployment for the `r7i` rerun.

## Server Setup

Validated deployment:

- Milvus image: `milvusdb/milvus:v2.6.17`.
- etcd image: `quay.io/coreos/etcd:v3.5.25`.
- MinIO image: `minio/minio:RELEASE.2024-12-18T13-15-44Z`.
- Deployment: official `milvus-standalone-docker-compose.yml`.
- MQ config: `MQ_TYPE=woodpecker` from the official `v2.6.17` compose file.
- Persistent data: `~/milvus-standalone/volumes`.

Reproducible fresh-deploy script:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

export MILVUS_VERSION=v2.6.17
wget "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml" \
  -O docker-compose.yml

sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "300m",
    "max-file": "3"
  }
}
JSON

sudo systemctl restart docker
sudo docker compose pull
sudo docker compose down -v --remove-orphans || true
sudo rm -rf volumes
sudo docker compose up -d
sudo docker compose ps
curl -fsS http://127.0.0.1:9091/healthz
```

Fresh teardown script used after the `r7i` rerun:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/milvus-standalone
sudo docker compose down -v --remove-orphans
sudo rm -rf volumes
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

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
  --task-label "fts-e2e-milvus-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The stability rerun used the same command with task label `fts-e2e-milvus-msmarco-small-stability`.
The `r7i.4xlarge` rerun used the same command with task label `fts-e2e-milvus-msmarco-small-r7i`.
The `r7i.4xlarge` medium run used the same command with task label `fts-e2e-milvus-msmarco-medium-r7i` and dataset size `MS MARCO Medium (1M documents)`.

Exact client script for the `r7i.4xlarge` MS MARCO Small text-payload run:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
  --task-label fts-e2e-milvus-msmarco-small-text-r7i-rerun \
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

  python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
    --uri "http://${SERVER_HOST}:19530" \
    --task-label "fts-msmarco-medium-milvus-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "MS MARCO Medium (1M documents)" \
    --payload-profile "${PAYLOAD_PROFILE}" \
    --drop-old --load --search-serial --search-concurrent \
    --k 100 --concurrency-duration 30 \
    --num-concurrency "1,10,20,40,60,80" \
    --concurrency-timeout 3600
done
```

Effective Milvus FTS case config from the raw JSON:

- `index_type=SPARSE_INVERTED_INDEX`
- `metric_type=BM25`
- `inverted_index_algo=DAAT_MAXSCORE`
- `bm25_k1=1.5`
- `bm25_b=0.75`
- `analyzer_tokenizer=standard`
- `analyzer_enable_lowercase=true`
- `analyzer_max_token_length=40`
- `num_shards=1`
- `replica_number=1`

## Result

| Raw JSON | Task label | Dataset size | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/5/10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260528_fts-e2e-milvus-msmarco-small_milvus.json` | `fts-e2e-milvus-msmarco-small` | 100K | 248.2907 | 999.1071 | 0.9157 | 0.7157 | 0.6653 | 0.0099 | 0.0150 | 260.7754 / 357.2333 / 730.4332 / 999.1071 |
| `result_20260601_fts-e2e-milvus-msmarco-small_milvus.json` | `fts-e2e-milvus-msmarco-small` | 100K | 227.1237 | 4098.1460 | 0.9157 | 0.7157 | 0.6653 | 0.0022 | 0.0027 | 567.2801 / 2331.7462 / 3494.7072 / 4098.1460 |
| `result_20260601_fts-e2e-milvus-msmarco-small-stability_milvus.json` | `fts-e2e-milvus-msmarco-small-stability` | 100K | 233.5157 | 4135.7169 | 0.9157 | 0.7157 | 0.6653 | 0.0022 | 0.0027 | 566.9304 / 2317.6759 / 3526.9488 / 4135.7169 |
| `result_20260602_fts-e2e-milvus-msmarco-small-r7i_milvus.json` | `fts-e2e-milvus-msmarco-small-r7i` | 100K | 230.3305 | 9359.8351 | 0.9157 | 0.7157 | 0.6653 | 0.0026 | 0.0029 | 528.3714 / 3129.5306 / 5750.1304 / 9359.8351 |
| `result_20260602_fts-e2e-milvus-msmarco-medium-r7i_milvus.json` | `fts-e2e-milvus-msmarco-medium-r7i` | 1M | 2042.1265 | 3857.7069 | 0.8048 | 0.5174 | 0.4458 | 0.0056 | 0.0074 | 284.4606 / 1608.7846 / 2858.1176 / 3857.7069 |

Latest `r7i.4xlarge` rerun vs previous `m5d.2xlarge` stability run:

- QPS increased from `4135.7169` to `9359.8351` (+126.3%).
- Load duration changed from `233.5157s` to `230.3305s` (-1.4%).
- Recall stayed unchanged at `0.9157`.
- p95 changed from `0.0022s` to `0.0026s`; p99 changed from `0.0027s` to `0.0029s`.

MS MARCO Small text payload rerun on `r7i.4xlarge`:

| Raw JSON | Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260603_fts-e2e-milvus-msmarco-small-text-r7i-rerun_milvus.json` | `fts-e2e-milvus-msmarco-small-text-r7i-rerun` | 230.4392 | 9569.0676 | 0.9157 | 0.7157 | 0.6653 | 0.0029 | 0.0032 | 468.2255 / 4857.3898 / 8011.723 / 9279.3577 / 9569.0676 / 9266.8844 |

Text payload details:

- `payload_profile=text`.
- Returned fields: `doc_id` and `text`.
- Estimated payload bytes per query from VectorDBBench: `53200`.

MS MARCO Medium six-concurrency rerun on `r7i.4xlarge`:

| Raw JSON | Payload | Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260604_fts-msmarco-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T041648Z_milvus.json` | `ids_only` | `fts-msmarco-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T041648Z` | 2048.1231 | 5139.5920 | 0.8048 | 0.5174 | 0.4458 | 0.0053 | 0.0071 | 433.1311 / 2976.3670 / 3973.3733 / 4750.5123 / 5053.7822 / 5139.5920 |
| `result_20260604_fts-msmarco-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T041648Z_milvus.json` | `text` | `fts-msmarco-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T041648Z` | 2048.2360 | 4677.4078 | 0.8048 | 0.5174 | 0.4458 | 0.0057 | 0.0075 | 378.3115 / 2732.7863 / 3656.8277 / 4353.0352 / 4602.9011 / 4677.4078 |

MS MARCO Medium stability comparison against the previous `r7i.4xlarge` ids-only run:

- QPS increased from `3857.7069` to `5139.5920` (+33.2%).
- Load duration changed from `2042.1265s` to `2048.1231s` (+0.3%).
- Recall stayed unchanged at `0.8048`.
- Text payload QPS was `4677.4078`, which is 9.0% below the new ids-only rerun at the same concurrency list.
