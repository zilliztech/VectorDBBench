# Milvus HotpotQA FTS E2E Report

## Prequsites

- Backend: Milvus standalone.
- Dataset family: HotpotQA.
- Current committed raw results: `HotpotQA Medium (1M documents)` on the `r7i.4xlarge` server.
- Run dates represented here: 2026-06-02.
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

Exact client script for the committed HotpotQA Medium run:

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
| `result_20260602_fts-e2e-milvus-hotpotqa-medium-r7i_milvus.json` | `fts-e2e-milvus-hotpotqa-medium-r7i` | 1M | 2031.2796 | 1596.6340 | 0.8378 | 0.7246 | 0.8561 | 0.0123 | 0.0170 | 146.0629 / 726.2209 / 1273.3380 / 1596.6340 |
