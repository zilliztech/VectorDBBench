# Milvus FTS End-to-End Runbook

This runbook validates the Milvus BM25 full-text-search backend against a
remote Milvus standalone deployment. Keep this file Milvus-specific; shared
dataset, metric, and backend matrix decisions belong in
`docs/fts-e2e-testing-plan.md`.

## Topology

Use two machines when possible:

- Server: runs Milvus standalone with Docker Compose.
- Client: runs VectorDBBench and stores datasets/results.

The tested baseline used private EC2 networking with the client connecting to
Milvus proxy port `19530` and health port `9091`. Only `19530` is required for
VectorDBBench. Keep etcd and MinIO private.

## Server Deployment

Install Docker and Docker Compose on the server. The validated baseline used:

- Docker `24.0.5`
- Docker Compose `v2.27.0`
- Milvus image `milvusdb/milvus:v2.6.17`
- etcd image `quay.io/coreos/etcd:v3.5.25`
- MinIO image `minio/minio:RELEASE.2024-12-18T13-15-44Z`

Deploy the official standalone compose file:

```bash
mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

export MILVUS_VERSION=v2.6.17
wget "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml" \
  -O docker-compose.yml

sudo docker compose pull
sudo docker compose up -d
sudo docker compose ps
```

Important compose details from the validated baseline:

- `standalone` runs `milvusdb/milvus:v2.6.17`.
- `MQ_TYPE` is `woodpecker`, as set by the official `v2.6.17` compose file.
- `standalone` exposes `19530` and `9091`.
- MinIO exposes `9000` and `9001` in the official compose file; do not expose
  those publicly unless you explicitly need them.
- Persistent data is under `~/milvus-standalone/volumes`.

Configure Docker log rotation before longer runs:

```bash
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
cd ~/milvus-standalone
sudo docker compose up -d
```

Health checks on the server:

```bash
cd ~/milvus-standalone
sudo docker compose ps
curl -fsS http://127.0.0.1:9091/healthz
sudo docker logs --tail 100 milvus-standalone
df -h /
```

Expected health output:

```text
OK
```

To reset the server for a clean run:

```bash
cd ~/milvus-standalone
sudo docker compose down -v
sudo docker compose up -d
```

## Client Setup

Run all VectorDBBench commands from the repository root on the client.

```bash
cd /home/ubuntu/VectorDBBench
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
```

The validated baseline used branch `fts`, commit `21a68b1`.

The client Python environment must use `pydantic<2`. The validated EC2 client
already had Pydantic 2 globally, so it used a temporary Pydantic v1 target:

```bash
export PYTHONPATH=/tmp/vdbbench-pydantic-v1:/home/ubuntu/VectorDBBench
```

In a clean environment, prefer installing the project dependencies so this
workaround is unnecessary.

Set benchmark paths and Milvus URI:

```bash
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

export MILVUS_HOST="<milvus-server-private-ip-or-dns>"
export MILVUS_URI="http://${MILVUS_HOST}:19530"
export MILVUS_USER=""
export MILVUS_PASSWORD=""
```

The current `milvusfts` command does not expose TLS certificate options. Use
plaintext only on a private network, VPN, or SSH tunnel unless TLS support is
added later. If Milvus auth is disabled, omit username/password flags rather
than passing empty values.

## Client Preflight

Run focused FTS tests before the remote E2E run:

```bash
python3.11 -m pytest \
  tests/test_fts_dataset.py \
  tests/test_fts_cases.py \
  tests/test_fts_runners.py \
  tests/test_fts_backend_capability.py \
  tests/test_fts_milvus.py \
  tests/test_fts_format_results.py \
  -q
```

The validated baseline passed the focused FTS suite with `86 passed`.

Verify network and PyMilvus connectivity from the client:

```bash
curl -fsS "http://${MILVUS_HOST}:9091/healthz"
nc -vz "$MILVUS_HOST" 19530

python3.11 - <<'PY'
import os
from pymilvus import connections, utility

connections.connect(uri=os.environ["MILVUS_URI"], timeout=10)
print("server_version:", utility.get_server_version())
print("collections:", utility.list_collections())
connections.disconnect("default")
PY
```

Expected baseline output:

```text
OK
server_version: 2.6.17
collections: []
```

Check disk before running. MS MARCO Small consumed about 3 GiB during dataset
preparation, but this client already had a large shared dataset cache. Keep at
least 20 GiB free for a small run and substantially more for medium/large runs:

```bash
df -h / /tmp
du -sh "$DATASET_LOCAL_DIR" 2>/dev/null || true
```

## Run MS MARCO Small

Use tmux so the run survives SSH disconnects:

```bash
RUN_ID="fts_milvus_small_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/home/ubuntu/bench-runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

tmux new-session -d -s "$RUN_ID" "bash -lc '
set -o pipefail
cd /home/ubuntu/VectorDBBench
export PYTHONPATH=/tmp/vdbbench-pydantic-v1:/home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export MILVUS_URI=http://${MILVUS_HOST}:19530

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri \"\$MILVUS_URI\" \
  --task-label fts-e2e-milvus-msmarco-small \
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

If Milvus auth is enabled, add:

```bash
--user-name "$MILVUS_USER" --password "$MILVUS_PASSWORD"
```

## Monitor the Run

Client-side progress:

```bash
tail -n 160 "$RUN_DIR/run.log"
test -f "$RUN_DIR/status" && cat "$RUN_DIR/status" || echo RUNNING
tmux list-sessions | grep "$RUN_ID" || true
df -h / /tmp
du -sh "$DATASET_LOCAL_DIR" 2>/dev/null || true
```

Server-side resource and health checks:

```bash
cd ~/milvus-standalone
sudo docker stats --no-stream \
  --format "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} net={{.NetIO}} block={{.BlockIO}}" \
  milvus-standalone milvus-etcd milvus-minio

curl -fsS http://127.0.0.1:9091/healthz
sudo docker logs --since 3m milvus-standalone | grep -Ei "error|fatal|panic|oom" | tail -50 || true
df -h /
```

Optional collection count check from the client:

```bash
python3.11 - <<'PY'
from pymilvus import connections, Collection, utility

connections.connect(uri="http://<milvus-server-private-ip-or-dns>:19530", timeout=10)
print("has_collection", utility.has_collection("VDBBench"))
if utility.has_collection("VDBBench"):
    col = Collection("VDBBench")
    print("num_entities", col.num_entities)
    print("indexes", [idx.index_name for idx in col.indexes])
connections.disconnect("default")
PY
```

Expected after load:

```text
has_collection True
num_entities 100000
indexes ['doc_id_sort_idx', 'sparse_vector_idx']
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
find "$RESULTS_LOCAL_DIR/Milvus" -maxdepth 1 -type f | sort | tail -20
```

Expected path pattern:

```text
/tmp/vectordb_bench/results/Milvus/result_YYYYMMDD_fts-e2e-milvus-msmarco-small_milvus.json
```

Read the saved artifact with the project model:

```bash
export RESULT_PATH=/tmp/vectordb_bench/results/Milvus/result_YYYYMMDD_fts-e2e-milvus-msmarco-small_milvus.json

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

- Milvus standalone `v2.6.17`.
- Official `milvus-standalone-docker-compose.yml`.
- Client command: `milvusfts` over `MS MARCO Small (100K documents)`.
- Result file:
  `/tmp/vectordb_bench/results/Milvus/result_20260528_fts-e2e-milvus-msmarco-small_milvus.json`.
- Run ID: `50f9f27a4be941849b40b12511fff9b5`.
- Result label: `ResultLabel.NORMAL`.

Metrics:

```text
insert_duration: 218.0799s
optimize_duration: 30.2108s
load_duration: 248.2907s
qps: 999.1071
serial_latency_p99: 0.0150s
serial_latency_p95: 0.0099s
recall: 0.9157
ndcg: 0.7157
mrr: 0.6653
conc_num_list: [1, 5, 10, 20]
conc_qps_list: [260.7754, 357.2333, 730.4332, 999.1071]
conc_latency_p99_list: [0.023011206425144332, 0.040039328668208356, 0.04286585776950232, 0.06397024502337442]
conc_latency_p95_list: [0.01070283028820995, 0.029407395451562467, 0.028955823008436708, 0.04165105621650582]
conc_latency_avg_list: [0.003831698230199239, 0.01397923275509955, 0.013673856742792125, 0.01998730334608544]
```

## Troubleshooting

Pydantic import errors:

- Symptom: model validation/import errors because the environment uses
  Pydantic 2.
- Fix: install project dependencies with `pydantic<2` or use a temporary
  Pydantic v1 target in `PYTHONPATH`.

Client disk pressure:

- Symptom: dataset preparation stalls or fails while `ir_datasets` extracts or
  fixes encoding.
- Fix: free space under `/tmp`, move `DATASET_LOCAL_DIR` to a larger disk, or
  remove old unused dataset caches.

Milvus is reachable on `9091` but not `19530`:

- Symptom: health check succeeds but PyMilvus cannot connect.
- Fix: open `19530` in the private security group or use SSH tunneling.

Insert is active but the client log is quiet:

- The first draft logs start/end of FTS insert but not every batch. Check server
  `docker stats` network/CPU and use the collection count script after a few
  minutes.

Server warnings during the validated run:

- Woodpecker WAL slow append warnings appeared under insert load.
- Sparse inverted index config warnings appeared during load.
- Neither warning failed the baseline run. Treat new `ERROR`, `FATAL`, `PANIC`,
  OOM, or repeated health failures as blockers.
