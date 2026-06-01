# Vespa FTS End-to-End Runbook

This runbook validates the Vespa BM25 full-text-search backend through the
existing `vespa` VectorDBBench command. Keep this file Vespa-specific; shared
dataset, metric, and backend matrix decisions belong in
`docs/fts-e2e-testing-plan.md`.

## Topology

Use two machines when possible:

- Server: runs a single-node Vespa container.
- Client: runs VectorDBBench and stores datasets/results.

The validated baseline used private EC2 networking with the client connecting
to Vespa HTTP port `8080` and Vespa config/deploy port `19071`.

## Server Deployment

Install Docker on the server. The validated baseline used:

- Vespa image `vespaengine/vespa:8.694.53`
- One container named `vespa`
- Persistent host directories under `/srv/vespa/var` and `/srv/vespa/logs`
- Feed/query API on `0.0.0.0:8080`
- Config/deploy API on `0.0.0.0:19071`
- Self-hosted unauthenticated HTTP on a private benchmark network

Do not expose this unauthenticated deployment publicly.

```bash
sudo mkdir -p /srv/vespa/var /srv/vespa/logs
sudo chown -R 1000:1000 /srv/vespa/var /srv/vespa/logs

docker rm -f vespa >/dev/null 2>&1 || true
docker pull vespaengine/vespa:8.694.53

docker run -d --name vespa --user vespa:vespa --hostname vespa-container \
  --ulimit nofile=262144:262144 --pids-limit=-1 \
  -v /srv/vespa/var:/opt/vespa/var \
  -v /srv/vespa/logs:/opt/vespa/logs \
  -p 0.0.0.0:8080:8080 \
  -p 0.0.0.0:19071:19071 \
  --restart unless-stopped \
  vespaengine/vespa:8.694.53
```

Health checks on the server:

```bash
docker ps --filter name=vespa
curl -fsS http://127.0.0.1:19071/state/v1/health
curl -fsS http://127.0.0.1:8080/state/v1/health || true
docker logs --tail 200 vespa
```

A fresh Vespa container may report the config/deploy API as up while the `8080`
endpoint is not useful yet. The VDBBench Vespa adapter deploys the application
package at run start; after deployment, `8080/state/v1/health` should report
`"code" : "up"`.

## Server Teardown For Fresh Runs

Run this teardown before a fresh E2E benchmark when the Vespa server state must
not carry over from a previous run. It removes the Vespa container and clears
the persisted Vespa state/log directories. Only use this on a disposable
benchmark deployment.

```bash
docker rm -f vespa >/dev/null 2>&1 || true
sudo find /srv/vespa/var -mindepth 1 -delete
sudo find /srv/vespa/logs -mindepth 1 -delete

docker run -d --name vespa --user vespa:vespa --hostname vespa-container \
  --ulimit nofile=262144:262144 --pids-limit=-1 \
  -v /srv/vespa/var:/opt/vespa/var \
  -v /srv/vespa/logs:/opt/vespa/logs \
  -p 0.0.0.0:8080:8080 \
  -p 0.0.0.0:19071:19071 \
  --restart unless-stopped \
  vespaengine/vespa:8.694.53

curl -fsS http://127.0.0.1:19071/state/v1/health
```

## Client Setup

Run all VectorDBBench commands from the repository root on the client.

```bash
cd /home/ubuntu/VectorDBBench
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
```

The validated baseline used branch `fts`, commit `80070a5`, which includes the
Elasticsearch runbook commit on top of the FTS implementation.

Current FTS draft code is aligned with Pydantic 2 APIs. Do not export the old
temporary Pydantic v1 `PYTHONPATH` override when running this branch.

Set benchmark paths and the Vespa endpoint:

```bash
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

export VESPA_URI="http://<vespa-server-private-ip-or-dns>"
export VESPA_PORT=8080
```

Do not include a port in `VESPA_URI`. The VDBBench adapter uses
`VESPA_URI:19071` for application deployment and `VESPA_URI:VESPA_PORT` for
feed/query.

## Client Preflight

Verify network reachability from the client:

```bash
curl -fsS "${VESPA_URI}:19071/state/v1/health"
curl -fsS "${VESPA_URI}:${VESPA_PORT}/state/v1/health" || true
```

Run the focused Vespa FTS tests:

```bash
python3.11 -m pytest \
  tests/test_fts_vespa.py \
  tests/test_fts_backend_capability.py \
  tests/test_fts_cases.py \
  -q
```

The validated baseline passed:

```text
29 passed
```

Run the CLI dry-run:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --dry-run \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label fts-e2e-vespa-msmarco-small \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected dry-run config:

- `db=<DB.Vespa: 'Vespa'>`
- `db_case_config=VespaFtsConfig()`
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
RUN_ID="fts_vespa_small_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/home/ubuntu/bench-runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/run.sh" <<'RUN'
#!/usr/bin/env bash
set -o pipefail
cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label fts-e2e-vespa-msmarco-small \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600

rc=$?
echo "EXIT_CODE=$rc" > "$(dirname "$0")/status"
exit $rc
RUN
chmod +x "$RUN_DIR/run.sh"

tmux new-session -d -s "$RUN_ID" \
  "bash '$RUN_DIR/run.sh' > '$RUN_DIR/run.log' 2>&1"

echo "$RUN_ID"
echo "$RUN_DIR"
```

The validated baseline used:

```bash
VESPA_URI=http://10.15.15.56
VESPA_PORT=8080
```

## Monitor the Run

Client-side progress:

```bash
tail -n 220 "$RUN_DIR/run.log"
test -f "$RUN_DIR/status" && cat "$RUN_DIR/status" || echo RUNNING
tmux list-sessions | grep "$RUN_ID" || true
ls -lt /tmp/vectordb_bench/results/Vespa 2>/dev/null | head || true
```

Server-side resource and health checks:

```bash
docker stats --no-stream \
  --format "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} net={{.NetIO}} block={{.BlockIO}}" \
  vespa

curl -fsS http://127.0.0.1:19071/state/v1/health
curl -fsS http://127.0.0.1:8080/state/v1/health
curl -fsS --get "http://127.0.0.1:8080/search/" \
  --data-urlencode "yql=select id from VectorDBBenchCollection where true limit 1"
```

A successful load smoke query should show `"totalCount":100000` for the small
MS MARCO case.

## Validated Baseline

Run details:

- Date: 2026-05-28
- Client: `/home/ubuntu/VectorDBBench` on branch `fts`
- Server: `10.15.15.56`
- Vespa image: `vespaengine/vespa:8.694.53`
- Container: `vespa`
- Run session: `fts_vespa_small_20260528_102843`
- Run log: `/home/ubuntu/bench-runs/fts_vespa_small_20260528_102843/run.log`
- Result file: `/tmp/vectordb_bench/results/Vespa/result_20260528_fts-e2e-vespa-msmarco-small_vespa.json`
- Run ID: `89727970c73847a3bfb9917f3ad74065`
- Status: `EXIT_CODE=0`

Metrics:

- Insert duration: `171.9692s`
- Optimize duration: `0.0s`
- Load duration: `171.9693s`
- Best QPS: `478.5384`
- Recall: `0.9416`
- NDCG: `0.7509`
- MRR: `0.7015`
- Serial p99 latency: `0.0269s`
- Serial p95 latency: `0.0212s`
- Concurrent QPS by concurrency `[1, 5, 10, 20]`: `[63.4393, 146.3972, 428.5583, 478.5384]`
- Concurrent p99 latency: `[0.0409538384, 0.1226536669, 0.0605090858, 0.1087617362]`
- Concurrent p95 latency: `[0.0287809825, 0.0871651917, 0.0449881301, 0.0856646255]`

Baseline caveats:

- The host was not isolated. Milvus standalone, Elasticsearch, and Vespa were
  all running on the same server.
- A concurrent `/home/ec2-user/vecTool/new-nightly-runner/new_nightly.sh` build
  was active during the run, compiling Knowhere/Cardinal/FAISS with
  `make -C build -j8`.
- Treat this result as a functional/comparable first pass, not as a clean
  isolated performance number.

## Clean Rebench 2026-06-01

Run details:

- Client branch: `fts`, commit `dc90056` plus the local FTS payload-estimate
  fix for text datasets without vector dimensions.
- Focused Vespa FTS tests:
  `python3.11 -m pytest tests/test_fts_vespa.py tests/test_fts_backend_capability.py tests/test_fts_cases.py -q`
  passed with `30 passed`.
- Server cleanup before rerun:
  `docker rm -f vespa`, `sudo find /srv/vespa/var -mindepth 1 -delete`, and
  `sudo find /srv/vespa/logs -mindepth 1 -delete`.
- Server state: only the Vespa benchmark container was running for this pass.
- Vespa image: `vespaengine/vespa:8.694.53`.
- Client command: `vespa` over `MS MARCO Small (100K documents)`.
- Pydantic note: do not export the old Pydantic v1 `PYTHONPATH`; the current
  branch uses Pydantic 2 APIs.
- Run session: `fts_vespa_small_20260601_103230`.
- Run log:
  `/home/ubuntu/bench-runs/fts_vespa_small_20260601_103230/run.log`.
- Status file:
  `/home/ubuntu/bench-runs/fts_vespa_small_20260601_103230/status`
  contained `EXIT_CODE=0`.
- Result file:
  `/tmp/vectordb_bench/results/Vespa/result_20260601_fts-e2e-vespa-msmarco-small_vespa.json`.
- Run ID: `01799f3c154c45269146d299179b347f`.
- Result label: `ResultLabel.NORMAL`.
- Post-load smoke query returned `"totalCount":100000`.

Metrics:

```text
insert_duration: 81.6402s
optimize_duration: 0.0s
load_duration: 81.6402s
qps: 482.1899
serial_latency_p99: 0.0260s
serial_latency_p95: 0.0202s
recall: 0.9416
ndcg: 0.7509
mrr: 0.7015
payload_profile: ids_only
payload_estimated_bytes_per_query: 2000
conc_num_list: [1, 5, 10, 20]
conc_qps_list: [78.8898, 336.8178, 482.1899, 470.0384]
conc_latency_p99_list: [0.027409066238906255, 0.03436350086471066, 0.04910965925082566, 0.11733379138866437]
conc_latency_p95_list: [0.02164906209800392, 0.027666533133015033, 0.03846237703692168, 0.0902713987394236]
conc_latency_avg_list: [0.012674093213950232, 0.014840469867348351, 0.02072369960953106, 0.042517835420612375]
```
