# ElasticSearch HotpotQA FTS E2E Report

## Prequsites

- Backend: Elasticsearch single-node container, invoked through VectorDBBench `elasticcloudhnsw`.
- Dataset family: HotpotQA.
- Current committed raw results: `HotpotQA Medium (1M documents)`, historical `HotpotQA Large (5.2M documents)`, and a `HotpotQA Large (5.2M documents)` text-payload matrix run on the `r7i.4xlarge` server.
- Run dates represented here: 2026-06-02 through 2026-06-04.
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

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Elasticsearch container.

## Server Setup

Validated deployment:

- Elasticsearch image: `docker.elastic.co/elasticsearch/elasticsearch:8.16.0`.
- Container name: `es01`.
- Docker memory limit: `-m 8g`.
- JVM heap: no explicit `ES_JAVA_OPTS` is set in the current baseline; Elasticsearch 8 auto-sizes from the container limit.
- JVM heap confirmed on the `r7i` run: `-Xms4096m -Xmx4096m`.
- Docker volume: `esdata01`.
- Security: disabled for isolated private benchmark networking.

Reproducible fresh-deploy script:

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

Fresh teardown script used after the run:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f es01
sudo docker volume rm esdata01
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

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "${SERVER_HOST}" \
  --port "9200" \
  --task-label "fts-e2e-elastic-hotpotqa-medium-r7i" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The committed HotpotQA Large run used the same command with task label `fts-e2e-elastic-hotpotqa-large-r7i` and dataset size `HotpotQA Large (5.2M documents)`.

Exact client script for the 2026-06-04 HotpotQA Large text-payload matrix run:

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
  --task-label "fts-matrix-elastic-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
  --payload-profile text \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "20,40,80" \
  --concurrency-timeout 3600
```

Effective Elasticsearch FTS case config from the raw JSON:

- `number_of_shards=1`
- `number_of_replicas=0`
- `refresh_interval=30s`
- `use_force_merge=true`
- `use_ssl=false`
- `verify_certs=true`

## Result

The ids-only matrix run `fts-matrix-elastic-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z` is intentionally excluded from the result table because VDBBench emitted only a zero-metric failure placeholder JSON. Log evidence shows the run loaded successfully (`load_duration=545.7043s`) and completed concurrency 20/40 (`447.5993 / 480.0184 QPS`), but the parent process hung after starting concurrency 80 and was terminated with `RUN_FAILED_143`.

| Raw JSON | Task label | Dataset size | Payload | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `result_20260602_fts-e2e-elastic-hotpotqa-medium-r7i_elasticcloud.json` | `fts-e2e-elastic-hotpotqa-medium-r7i` | 1M | ids_only | 142.0589 | 1410.3787 | 0.8378 | 0.7287 | 0.8598 | 0.0159 | 0.0224 | 1/5/10/20 | 111.5252 / 558.2304 / 1034.1176 / 1410.3787 |
| `result_20260602_fts-e2e-elastic-hotpotqa-large-r7i_elasticcloud.json` | `fts-e2e-elastic-hotpotqa-large-r7i` | 5.2M | ids_only | 550.6164 | 476.2610 | 0.7637 | 0.6243 | 0.7549 | 0.0503 | 0.0755 | 1/5/10/20 | 41.0129 / 202.7703 / 356.3845 / 476.2610 |
| `result_20260604_fts-matrix-elastic-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z_elasticcloud.json` | `fts-matrix-elastic-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z` | 5.2M | text | 554.4492 | 435.1027 | 0.7637 | 0.6243 | 0.7549 | 0.0518 | 0.0766 | 20/40/80 | 402.3090 / 435.1027 / 434.3993 |
