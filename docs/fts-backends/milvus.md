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

## Server Teardown For Fresh Runs

Run this teardown before a fresh E2E benchmark when the Milvus server state
must not carry over from a previous run. It removes Milvus, etcd, and MinIO
containers plus their persisted benchmark data under the standalone compose
directory. Only use this on a disposable benchmark deployment.

```bash
cd ~/milvus-standalone
sudo docker compose down -v --remove-orphans
sudo rm -rf volumes

sudo docker compose up -d
sudo docker compose ps
curl -fsS http://127.0.0.1:9091/healthz
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

Current FTS draft code is aligned with Pydantic 2 APIs. Do not export the old
temporary Pydantic v1 `PYTHONPATH` override when running this branch.

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

## Clean Rebench 2026-06-01

Run details:

- Client branch: `fts`, commit `dc90056` plus the local FTS payload-estimate
  fix for text datasets without vector dimensions.
- Focused FTS tests before rerun:
  `python3.11 -m pytest tests/test_fts_dataset.py tests/test_fts_cases.py tests/test_fts_runners.py tests/test_fts_backend_capability.py tests/test_fts_milvus.py tests/test_fts_format_results.py -q`
  passed with `46 passed`.
- Server cleanup before rerun:
  `sudo docker compose down -v --remove-orphans` and `sudo rm -rf volumes`,
  followed by `sudo docker compose up -d`.
- Server state: only the Milvus standalone stack was running for this pass.
- Server image: `milvusdb/milvus:v2.6.17`.
- Client command: `milvusfts` over `MS MARCO Small (100K documents)`.
- Pydantic note: do not export the old Pydantic v1 `PYTHONPATH`; the current
  branch uses Pydantic 2 APIs.
- Run session: `fts_milvus_small_20260601_100900`.
- Run log:
  `/home/ubuntu/bench-runs/fts_milvus_small_20260601_100900/run.log`.
- Status file:
  `/home/ubuntu/bench-runs/fts_milvus_small_20260601_100900/status`
  contained `EXIT_CODE=0`.
- Result file:
  `/tmp/vectordb_bench/results/Milvus/result_20260601_fts-e2e-milvus-msmarco-small_milvus.json`.
- Run ID: `6a330e8d3fd14fd9852193322e19de8d`.
- Result label: `ResultLabel.NORMAL`.

Metrics:

```text
insert_duration: 216.5893s
optimize_duration: 10.5344s
load_duration: 227.1237s
qps: 4098.1460
serial_latency_p99: 0.0027s
serial_latency_p95: 0.0022s
recall: 0.9157
ndcg: 0.7157
mrr: 0.6653
payload_profile: ids_only
payload_estimated_bytes_per_query: 2000
conc_num_list: [1, 5, 10, 20]
conc_qps_list: [567.2801, 2331.7462, 3494.7072, 4098.146]
conc_latency_p99_list: [0.0026880667638033615, 0.0035670675651635992, 0.005668817160185418, 0.01087720695184544]
conc_latency_p95_list: [0.00221023999620229, 0.0027692234027199445, 0.0041898298542946575, 0.008188851265003901]
conc_latency_avg_list: [0.0017611498639764328, 0.0021415530623254057, 0.002856634355362644, 0.004868861634112188]
```

## Stability Rerun 2026-06-01

The clean result above was repeated once with the same fresh teardown procedure:
`sudo docker compose down -v --remove-orphans`, `sudo rm -rf volumes`, and a
fresh `sudo docker compose up -d`.

Run details:

- Run session: `fts_milvus_small_stability_20260601_115836`.
- Run log:
  `/home/ubuntu/bench-runs/fts_milvus_small_stability_20260601_115836/run.log`.
- Status file:
  `/home/ubuntu/bench-runs/fts_milvus_small_stability_20260601_115836/status`
  contained `EXIT_CODE=0`.
- Result file:
  `/tmp/vectordb_bench/results/Milvus/result_20260601_fts-e2e-milvus-msmarco-small-stability_milvus.json`.
- Run ID: `1981d8ea3c7f4282b430c7b3ef1c7115`.
- Result label: `ResultLabel.NORMAL`.

Metrics:

```text
insert_duration: 215.9478s
optimize_duration: 17.5679s
load_duration: 233.5157s
qps: 4135.7169
serial_latency_p99: 0.0027s
serial_latency_p95: 0.0022s
recall: 0.9157
ndcg: 0.7157
mrr: 0.6653
payload_profile: ids_only
payload_estimated_bytes_per_query: 2000
conc_num_list: [1, 5, 10, 20]
conc_qps_list: [566.9304, 2317.6759, 3526.9488, 4135.7169]
conc_latency_p99_list: [0.0027008821396157156, 0.003622065201634541, 0.005597269621212032, 0.010705483118072156]
conc_latency_p95_list: [0.0022123721195384857, 0.0027939058432821184, 0.004106056515593081, 0.008097446372266857]
conc_latency_avg_list: [0.001762315369339218, 0.0021546111395611333, 0.002830934044328618, 0.004825849693942325]
```

Stability observation:

- Best QPS changed from `4098.1460` to `4135.7169`, about `+0.92%`.
- Recall, NDCG, MRR, p99, and p95 matched the clean run after rounding.
- This confirms the clean Milvus result is stable for this MS MARCO Small setup.

## Troubleshooting

Pydantic import errors:

- Symptom: import errors such as missing `field_validator` or `model_validator`
  when the old temporary Pydantic v1 `PYTHONPATH` target is exported.
- Fix: unset the temporary Pydantic v1 `PYTHONPATH` override and run with the
  current branch dependencies.

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
