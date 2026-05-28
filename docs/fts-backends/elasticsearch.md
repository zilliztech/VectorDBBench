# Elasticsearch FTS End-to-End Runbook

This runbook validates the Elasticsearch-compatible BM25 full-text-search
backend through the existing `elasticcloudhnsw` VectorDBBench command. Keep this
file Elasticsearch-specific; shared dataset, metric, and backend matrix
decisions belong in `docs/fts-e2e-testing-plan.md`.

## Topology

Use two machines when possible:

- Server: runs a single-node Elasticsearch container.
- Client: runs VectorDBBench and stores datasets/results.

The validated baseline used private EC2 networking with the client connecting
to Elasticsearch HTTP port `9200`. Port `9300` is the Elasticsearch transport
port and is not required for this single-node benchmark.

## Server Deployment

Install Docker on the server. The validated baseline used:

- Elasticsearch image `docker.elastic.co/elasticsearch/elasticsearch:8.16.0`
- Single-node mode
- One data volume named `esdata01`
- HTTP on `0.0.0.0:9200`
- Security disabled for an isolated private benchmark network

Do not expose this security-disabled deployment publicly.

```bash
sudo sysctl -w vm.max_map_count=1048576

sudo docker rm -f es01 >/dev/null 2>&1 || true
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
```

Health checks on the server:

```bash
curl -fsS "http://127.0.0.1:9200/"
curl -fsS "http://127.0.0.1:9200/_cluster/health?pretty&wait_for_status=yellow&timeout=60s"
curl -fsS "http://127.0.0.1:9200/_cat/indices?v"
sudo docker logs --tail 100 es01
df -h /
```

Expected health status after startup:

```text
green
```

## Server Teardown For Fresh Runs

Run this teardown before a fresh E2E benchmark when the Elasticsearch server
state must not carry over from a previous run. It removes the benchmark
container and the `esdata01` Docker volume. Only use this on a disposable
benchmark deployment.

```bash
sudo docker rm -f es01 >/dev/null 2>&1 || true
sudo docker volume rm esdata01 >/dev/null 2>&1 || true
sudo docker volume create esdata01

sudo docker run -d --name es01 \
  -p 0.0.0.0:9200:9200 \
  --restart unless-stopped \
  --ulimit nofile=65535:65535 \
  -m 8g \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -v esdata01:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.16.0

curl -fsS "http://127.0.0.1:9200/_cluster/health?pretty&wait_for_status=yellow&timeout=60s"
```

## Client Setup

Run all VectorDBBench commands from the repository root on the client.

```bash
cd /home/ubuntu/VectorDBBench
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
```

The validated baseline used branch `fts`, commit `ab0f5a2`, which includes the
Milvus runbook commit on top of FTS implementation commit `21a68b1`.

The client Python environment must use `pydantic<2`. The validated EC2 client
had Pydantic 2 globally, so it used a temporary Pydantic v1 target:

```bash
export PYTHONPATH=/tmp/vdbbench-pydantic-v1:/home/ubuntu/VectorDBBench
```

Set benchmark paths and Elasticsearch endpoint:

```bash
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

export ELASTIC_HOST="<elasticsearch-server-private-ip-or-dns>"
export ELASTIC_PORT=9200
```

If a secure Elasticsearch deployment is used instead of the private
security-disabled baseline, add the relevant CLI flags:

```bash
--user-name "$ELASTIC_USER" --password "$ELASTIC_PASSWORD"
--use-ssl
--no-verify-certs
```

Use `--no-verify-certs` only for local E2E testing with a private CA or
self-signed certificate.

## Client Preflight

Verify network and Elasticsearch health from the client:

```bash
curl -fsS "http://${ELASTIC_HOST}:${ELASTIC_PORT}/"
curl -fsS "http://${ELASTIC_HOST}:${ELASTIC_PORT}/_cluster/health?pretty"
```

Run the focused Elasticsearch FTS test:

```bash
python3.11 -m pytest tests/test_fts_elastic_cloud.py -q
```

The validated baseline passed:

```text
16 passed
```

Run the CLI dry-run:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --dry-run \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --task-label fts-e2e-elastic-msmarco-small \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected dry-run config:

- `db=<DB.ElasticCloud: 'ElasticCloud'>`
- `db_case_config=ElasticCloudFtsConfig(...)`
- `case_id=<CaseType.FTSmsmarcoPerformance: 503>`
- stages include `drop_old`, `load`, `search_serial`, `search_concurrent`

Check disk before running:

```bash
df -h / /tmp
du -sh "$DATASET_LOCAL_DIR" 2>/dev/null || true
```

## Run MS MARCO Small

Use tmux so the run survives SSH disconnects:

```bash
RUN_ID="fts_elastic_small_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/home/ubuntu/bench-runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

tmux new-session -d -s "$RUN_ID" "bash -lc '
set -o pipefail
cd /home/ubuntu/VectorDBBench
export PYTHONPATH=/tmp/vdbbench-pydantic-v1:/home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host \"${ELASTIC_HOST}\" \
  --port \"${ELASTIC_PORT}\" \
  --task-label fts-e2e-elastic-msmarco-small \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type \"MS MARCO Small (100K documents)\" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency \"1,5,10,20\" \
  --concurrency-timeout 3600

rc=\$?
echo \"EXIT_CODE=\$rc\" > \"${RUN_DIR}/status\"
exit \$rc
' > '${RUN_DIR}/run.log' 2>&1"

echo "$RUN_ID"
echo "$RUN_DIR"
```

The validated baseline used:

```bash
--host 10.15.15.56
--port 9200
```

## Monitor the Run

Client-side progress:

```bash
tail -n 160 "$RUN_DIR/run.log"
test -f "$RUN_DIR/status" && cat "$RUN_DIR/status" || echo RUNNING
tmux list-sessions | grep "$RUN_ID" || true
df -h / /tmp
```

Server-side resource and health checks:

```bash
sudo docker stats --no-stream \
  --format "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} net={{.NetIO}} block={{.BlockIO}}" \
  es01

curl -fsS "http://127.0.0.1:9200/_cluster/health?pretty"
curl -fsS "http://127.0.0.1:9200/_cat/indices?v"
sudo docker logs --since 3m es01 | grep -Ei "ERROR|fatal|panic|exception|OutOfMemory" | tail -50 || true
df -h /
```

Expected after load:

```text
index: vdb_bench_indice
docs.count: 100000
replicas: 0
health: green
```

## Validate Results

Check the status file:

```bash
cat "$RUN_DIR/status"
```

Expected:

```text
EXIT_CODE=0
```

Find the result JSON:

```bash
find "$RESULTS_LOCAL_DIR/ElasticCloud" -maxdepth 1 -type f | sort | tail -20
```

Expected path pattern:

```text
/tmp/vectordb_bench/results/ElasticCloud/result_YYYYMMDD_fts-e2e-elastic-msmarco-small_elasticcloud.json
```

Read the saved artifact with the project model:

```bash
export RESULT_PATH=/tmp/vectordb_bench/results/ElasticCloud/result_YYYYMMDD_fts-e2e-elastic-msmarco-small_elasticcloud.json

python3.11 - <<'PY'
import os
from pathlib import Path
from vectordb_bench.models import TestResult

path = Path(os.environ["RESULT_PATH"])
result = TestResult.read_file(path)
case = result.results[0]

print("path:", path)
print("run_id:", result.run_id)
print("task_label:", result.task_label)
print("result_count:", len(result.results))
print("case_label:", case.label)
print("metrics:", case.metrics)
PY
```

## Validated Baseline

Validated on 2026-05-28 with two EC2 instances:

- Elasticsearch `8.16.0`.
- Single-node Docker deployment with security disabled on private networking.
- Client command: `elasticcloudhnsw` over `MS MARCO Small (100K documents)`.
- Result file:
  `/tmp/vectordb_bench/results/ElasticCloud/result_20260528_fts-e2e-elastic-msmarco-small_elasticcloud.json`.
- Run ID: `c949c7ee123e4c8589e29a0b82256d55`.
- Result label: `ResultLabel.NORMAL`.

Metrics:

```text
insert_duration: 41.4521s
optimize_duration: 30.9791s
load_duration: 72.4312s
qps: 1227.3373
serial_latency_p99: 0.0098s
serial_latency_p95: 0.0050s
recall: 0.9116
ndcg: 0.7159
mrr: 0.6665
conc_num_list: [1, 5, 10, 20]
conc_qps_list: [143.5116, 365.361, 672.9976, 1227.3373]
conc_latency_p99_list: [0.03934600790089456, 0.03579974054795457, 0.037561563240014986, 0.05568320580059669]
conc_latency_p95_list: [0.024235990742454305, 0.02974931430799188, 0.029752372499206096, 0.038332196010742337]
conc_latency_avg_list: [0.006966784008655295, 0.013679243169195, 0.01485311984437366, 0.016281856296064438]
```

## Milvus Comparison From Same Test Pass

The matching Milvus MS MARCO Small result from the same test pass:

```text
load_duration: 248.2907s
qps: 999.1071
recall: 0.9157
ndcg: 0.7157
mrr: 0.6653
serial_latency_p99: 0.0150s
serial_latency_p95: 0.0099s
conc_qps_list: [260.7754, 357.2333, 730.4332, 999.1071]
```

## Troubleshooting

Elasticsearch fails to start:

- Check `vm.max_map_count`; it should be at least `262144`.
- Check Docker memory limit. The validated baseline used `-m 8g`.
- Read `sudo docker logs --tail 200 es01`.

Client cannot connect:

- Verify the server security group or firewall allows private TCP `9200`.
- Check from the client with `curl http://${ELASTIC_HOST}:${ELASTIC_PORT}/`.

Indexing finishes but search is slow or unstable:

- Check `sudo docker stats es01` for CPU and memory pressure.
- Keep `number_of_shards=1` and `number_of_replicas=0` for the baseline.
- Keep `refresh_interval=30s` and `use_force_merge=True` unless comparing
  ingestion/search tradeoffs intentionally.

Unexpected benchmark result path:

- The command name is `elasticcloudhnsw`, but FTS output is still written under
  `$RESULTS_LOCAL_DIR/ElasticCloud`.
- The file suffix is `_elasticcloud.json`.
