# Milvus HotpotQA FTS E2E Report

## Prequsites

- Backend: Milvus standalone.
- Dataset family: HotpotQA.
- Current committed raw results: `HotpotQA Medium (1M documents)`, historical `HotpotQA Large (5.2M documents)`, and `HotpotQA Large (5.2M documents)` matrix runs with ids-only and text payloads on the `r7i.4xlarge` server.
- Run dates represented here: 2026-06-02, 2026-06-03, and 2026-06-04.
- Source runbook: `docs/fts-backends/milvus.md`.
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
- Role: runs the Milvus standalone server deployment.

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

Fresh teardown script used after the run:

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

Exact client script for the committed HotpotQA runs:

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
  --task-label "fts-e2e-milvus-hotpotqa-medium-r7i" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The committed HotpotQA Large run used the same command with task label `fts-e2e-milvus-hotpotqa-large-r7i` and dataset size `HotpotQA Large (5.2M documents)`.

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

  python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
    --uri "http://${SERVER_HOST}:19530" \
    --task-label "fts-hotpotqa-medium-milvus-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
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

Exact client script for the 2026-06-03 HotpotQA Large matrix runs:

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
  --task-label "fts-matrix-milvus-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "20,40,80" \
  --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
  --task-label "fts-matrix-milvus-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
  --payload-profile text \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "20,40,80" \
  --concurrency-timeout 3600
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

| Raw JSON | Task label | Dataset size | Payload | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `result_20260602_fts-e2e-milvus-hotpotqa-medium-r7i_milvus.json` | `fts-e2e-milvus-hotpotqa-medium-r7i` | 1M | ids_only | 2031.2796 | 1596.6340 | 0.8378 | 0.7246 | 0.8561 | 0.0123 | 0.0170 | 1/5/10/20 | 146.0629 / 726.2209 / 1273.3380 / 1596.6340 |
| `result_20260604_fts-hotpotqa-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T074646Z_milvus.json` | `fts-hotpotqa-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | ids_only | 2040.9336 | 1865.4681 | 0.8378 | 0.7246 | 0.8561 | 0.0122 | 0.0170 | 1/10/20/40/60/80 | 255.0087 / 1364.3522 / 1378.9975 / 1702.7745 / 1851.3955 / 1865.4681 |
| `result_20260604_fts-hotpotqa-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T074646Z_milvus.json` | `fts-hotpotqa-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | text | 2033.2594 | 1714.0357 | 0.8378 | 0.7246 | 0.8561 | 0.0124 | 0.0170 | 1/10/20/40/60/80 | 223.9828 / 1224.1637 / 1558.9074 / 1669.1467 / 1687.2785 / 1714.0357 |
| `result_20260602_fts-e2e-milvus-hotpotqa-large-r7i_milvus.json` | `fts-e2e-milvus-hotpotqa-large-r7i` | 5.2M | ids_only | 10583.8485 | 394.4417 | 0.7573 | 0.6129 | 0.7410 | 0.0212 | 0.0299 | 1/5/10/20 | 88.1695 / 336.8579 / 388.2553 / 394.4417 |
| `result_20260603_fts-matrix-milvus-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z_milvus.json` | `fts-matrix-milvus-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z` | 5.2M | ids_only | 10583.8402 | 411.7323 | 0.7573 | 0.6129 | 0.7410 | 0.0211 | 0.0305 | 20/40/80 | 400.7550 / 407.1847 / 411.7323 |
| `result_20260603_fts-matrix-milvus-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z_milvus.json` | `fts-matrix-milvus-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z` | 5.2M | text | 10583.7873 | 409.4366 | 0.7573 | 0.6129 | 0.7410 | 0.0214 | 0.0308 | 20/40/80 | 395.3148 / 407.5527 / 409.4366 |
