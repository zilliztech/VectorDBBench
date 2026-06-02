# ElasticSearch MS MARCO FTS E2E Report

## Prequsites

- Backend: Elasticsearch single-node container, invoked through VectorDBBench `elasticcloudhnsw`.
- Dataset family: MS MARCO.
- Current committed raw results: `MS MARCO Small (100K documents)`.
- Source runbook: `docs/fts-backends/elasticsearch.md`.
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

Server machine:

- EC2 type: `m5d.2xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 8 vCPU, Intel Xeon Platinum 8175M, 4 physical cores, 2 threads per core.
- Memory: about 30 GiB usable RAM plus 31 GiB swap.
- Disk quota: root XFS filesystem 500 GiB total with 451 GiB available at last check; additional 279 GiB NVMe mounted at `/data`.
- Docker: overlay2 under `/var/lib/docker` on the root filesystem.
- Role: runs the Elasticsearch container.

## Server Setup

Validated deployment:

- Elasticsearch image: `docker.elastic.co/elasticsearch/elasticsearch:8.16.0`.
- Container name: `es01`.
- Docker memory limit: `-m 8g`.
- JVM heap: no explicit `ES_JAVA_OPTS` is set in the current baseline; Elasticsearch 8 auto-sizes from the container limit.
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

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "${SERVER_HOST}" \
  --port "9200" \
  --task-label "fts-e2e-elastic-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The stability rerun used the same command with task label `fts-e2e-elastic-msmarco-small-stability`.

Effective Elasticsearch FTS case config from the raw JSON:

- `number_of_shards=1`
- `number_of_replicas=0`
- `refresh_interval=30s`
- `use_force_merge=true`
- `use_ssl=false`
- `verify_certs=true`

## Result

| Raw JSON | Task label | Dataset size | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/5/10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260528_fts-e2e-elastic-msmarco-small_elasticcloud.json` | `fts-e2e-elastic-msmarco-small` | 100K | 72.4312 | 1227.3373 | 0.9116 | 0.7159 | 0.6665 | 0.0050 | 0.0098 | 143.5116 / 365.3610 / 672.9976 / 1227.3373 |
| `result_20260601_fts-e2e-elastic-msmarco-small_elasticcloud.json` | `fts-e2e-elastic-msmarco-small` | 100K | 59.2031 | 3100.5973 | 0.9118 | 0.7159 | 0.6665 | 0.0031 | 0.0040 | 422.7782 / 1967.8125 / 2861.4449 / 3100.5973 |
| `result_20260601_fts-e2e-elastic-msmarco-small-stability_elasticcloud.json` | `fts-e2e-elastic-msmarco-small-stability` | 100K | 58.9212 | 3113.2707 | 0.9118 | 0.7159 | 0.6665 | 0.0031 | 0.0039 | 416.5153 / 1991.3322 / 2823.6226 / 3113.2707 |
