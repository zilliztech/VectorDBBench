# ElasticSearch HotpotQA FTS E2E Report

## Prequsites

- Backend: Elasticsearch single-node container, invoked through VectorDBBench `elasticcloudhnsw`.
- Dataset family: HotpotQA.
- Current committed raw results: none.
- Source runbook: `docs/fts-backends/elasticsearch.md`.
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
- Role: runs the Elasticsearch container.

## Server Setup

Use the same Elasticsearch setup as `../msmarco/report.md`: image `docker.elastic.co/elasticsearch/elasticsearch:8.16.0`, container `es01`, Docker memory limit `-m 8g`, no explicit `ES_JAVA_OPTS`, and a fresh `esdata01` volume before each run.

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo sysctl -w vm.max_map_count=1048576
sudo docker rm -f es01 >/dev/null 2>&1 || true
sudo docker volume rm esdata01 >/dev/null 2>&1 || true
sudo docker volume create esdata01
sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:8.16.0

sudo docker run -d --name es01 \
  -p 0.0.0.0:9200:9200 \
  --restart unless-stopped \
  --ulimit nofile=65535:65535 \
  -m 8g \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -v esdata01:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.16.0

curl -fsS "http://127.0.0.1:9200/_cluster/health?pretty&wait_for_status=yellow&timeout=90s"
```

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

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "${SERVER_HOST}" \
  --port "9200" \
  --task-label "fts-e2e-elastic-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected Elasticsearch FTS case config:

- `number_of_shards=1`
- `number_of_replicas=0`
- `refresh_interval=30s`
- `use_force_merge=true`
- `use_ssl=false`
- `verify_certs=true`

## Result

No HotpotQA Elasticsearch raw result JSON has been committed yet.
