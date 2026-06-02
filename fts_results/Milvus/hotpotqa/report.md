# Milvus HotpotQA FTS E2E Report

## Prequsites

- Backend: Milvus standalone.
- Dataset family: HotpotQA.
- Current committed raw results: none.
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

- EC2 type: `m5d.2xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 8 vCPU, Intel Xeon Platinum 8175M, 4 physical cores, 2 threads per core.
- Memory: about 30 GiB usable RAM plus 31 GiB swap.
- Disk quota: root XFS filesystem 500 GiB total with 451 GiB available at last check; additional 279 GiB NVMe mounted at `/data`.
- Docker: overlay2 under `/var/lib/docker` on the root filesystem.
- Role: runs the Milvus standalone server deployment.

## Server Setup

Use the same Milvus server setup as `../msmarco/report.md`: official Milvus standalone Docker Compose, `milvusdb/milvus:v2.6.17`, fresh `docker compose down -v`, and removal of `~/milvus-standalone/volumes` before each run.

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

export MILVUS_VERSION=v2.6.17
wget "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml" \
  -O docker-compose.yml

sudo docker compose pull
sudo docker compose down -v --remove-orphans || true
sudo rm -rf volumes
sudo docker compose up -d
sudo docker compose ps
curl -fsS http://127.0.0.1:9091/healthz
```

Milvus server config uses the official `v2.6.17` compose file, including `MQ_TYPE=woodpecker`.

## VDBBench Running

Reproducible client script for the planned HotpotQA Medium run:

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
  --task-label "fts-e2e-milvus-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected Milvus FTS case config:

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

No HotpotQA Milvus raw result JSON has been committed yet.
