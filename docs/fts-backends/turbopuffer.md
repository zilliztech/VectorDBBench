# TurboPuffer FTS End-to-End Runbook

This runbook validates the TurboPuffer BM25 full-text-search backend through
the existing `turbopuffer` VectorDBBench command. Keep this file
TurboPuffer-specific; shared dataset, metric, and backend matrix decisions
belong in `docs/fts-e2e-testing-plan.md`.

## Topology

TurboPuffer is a managed service. There is no server-side Docker deployment for
this backend. The client machine runs VectorDBBench and connects to the
TurboPuffer API over HTTPS.

Use one fresh namespace per E2E run:

```bash
export TURBOPUFFER_NAMESPACE="vdbbench-fts-tpuf-$(date -u +%Y%m%d-%H%M)-msmarco-small"
```

Do not expose the API key in command arguments, logs, or `ps` output. Prefer a
local config file with restrictive permissions, or pass the key through
interactive input when wrapping the VectorDBBench runner.

## Pricing Notes

Pricing checked on 2026-05-28:

- Storage: `$0.33/GB-month` on Launch/Scale.
- Writes: `$2/logical GB written`.
- Queries: `$0.001/GB processed` plus `$0.05/GB returned`.
- Query billing uses a minimum namespace size of `1.28 GB`.
- Plan minimums can dominate a small benchmark: Launch minimum `$64/month`,
  Scale minimum `$256/month`.

The MS MARCO Small run writes about 100K text rows. The measured namespace was
about 32.5 MB logical bytes after ingest, so the marginal storage/write/query
cost is small, but the account-level plan minimum is the real billing exposure
unless covered by existing committed usage.

## Client Setup

Run all VectorDBBench commands from the repository root on the client.

```bash
cd /home/ubuntu/VectorDBBench
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
```

The validated baseline used branch `fts`, commit `1ae680f`.

The client Python environment must use `pydantic<2`. The validated EC2 client
had Pydantic 2 globally, so it used a temporary Pydantic v1 target:

```bash
export PYTHONPATH=/tmp/vdbbench-pydantic-v1:/home/ubuntu/VectorDBBench
```

Set benchmark paths:

```bash
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export TURBOPUFFER_API_BASE_URL="https://api.turbopuffer.com"
export TURBOPUFFER_NAMESPACE="vdbbench-fts-tpuf-$(date -u +%Y%m%d-%H%M)-msmarco-small"
```

Create a config file for the API key instead of passing the key on the command
line:

```bash
TPUF_CONFIG="$(mktemp /tmp/turbopuffer-vdbbench.XXXXXX.yml)"
chmod 600 "$TPUF_CONFIG"

python3.11 - <<'PY' "$TPUF_CONFIG" "$TURBOPUFFER_NAMESPACE" "$TURBOPUFFER_API_BASE_URL"
import getpass
import sys

path, namespace, base_url = sys.argv[1:4]
api_key = getpass.getpass("TurboPuffer API key: ")
with open(path, "w", encoding="utf-8") as f:
    f.write("turbopuffer:\n")
    f.write(f"  api_key: {api_key}\n")
    f.write(f"  api_base_url: {base_url}\n")
    f.write(f"  namespace: {namespace}\n")
PY
```

Keep this file local to the benchmark host and remove it after the run.

## Client Preflight

Run the focused TurboPuffer FTS tests:

```bash
python3.11 -m pytest \
  tests/test_fts_turbopuffer.py \
  tests/test_fts_backend_capability.py \
  tests/test_fts_cases.py \
  -q
```

Run the CLI dry-run:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --config-file "$TPUF_CONFIG" \
  --dry-run \
  --task-label fts-e2e-tpuf-msmarco-small \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected dry-run config:

- `db=<DB.TurboPuffer: 'TurboPuffer'>`
- `db_case_config=TurboPufferFtsConfig(time_wait_warmup=60)`
- `case_id=<CaseType.FTSmsmarcoPerformance: 503>`
- stages include `drop_old`, `load`, `search_serial`, `search_concurrent`
- the API key is printed as `SecretStr('**********')`

## Run MS MARCO Small

Use tmux so the run survives SSH disconnects:

```bash
RUN_ID="fts_tpuf_small_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/home/ubuntu/bench-runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/run.sh" <<'RUN'
#!/usr/bin/env bash
set -o pipefail
cd /home/ubuntu/VectorDBBench
export PYTHONPATH=/tmp/vdbbench-pydantic-v1:/home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --config-file "$TPUF_CONFIG" \
  --task-label fts-e2e-tpuf-msmarco-small \
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
  "TPUF_CONFIG='$TPUF_CONFIG' bash '$RUN_DIR/run.sh' > '$RUN_DIR/run.log' 2>&1"

echo "$RUN_ID"
echo "$RUN_DIR"
```

Remove the config file after the run if it is not needed for teardown:

```bash
rm -f "$TPUF_CONFIG"
```

## Monitor the Run

Client-side progress:

```bash
tail -n 220 "$RUN_DIR/run.log"
test -f "$RUN_DIR/status" && cat "$RUN_DIR/status" || echo RUNNING
tmux list-sessions | grep "$RUN_ID" || true
ls -lt /tmp/vectordb_bench/results/TurboPuffer 2>/dev/null | head || true
```

Inspect namespace state without exposing the API key in process arguments:

```bash
python3.11 - <<'PY'
import getpass
import os
import turbopuffer as tpuf

api_key = getpass.getpass("TurboPuffer API key: ")
client = tpuf.Turbopuffer(
    api_key=api_key,
    base_url=os.environ.get("TURBOPUFFER_API_BASE_URL", "https://api.turbopuffer.com"),
)
ns = client.namespace(os.environ["TURBOPUFFER_NAMESPACE"])
print(ns.metadata())
PY
```

## Search-Only Rerun

If ingest completed but a later search phase failed, rerun search against the
same namespace without dropping or loading:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --config-file "$TPUF_CONFIG" \
  --task-label fts-e2e-tpuf-msmarco-small-searchonly \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --skip-drop-old --skip-load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The VectorDBBench search runner uses multiprocessing with spawn semantics. The
TurboPuffer SDK client contains non-picklable synchronization state, so the
VectorDBBench TurboPuffer adapter must exclude and recreate the SDK client when
the backend object is pickled for worker processes.

## Teardown For Fresh Runs

Run this teardown before a fresh E2E benchmark when the TurboPuffer namespace
must not carry over from a previous run:

```bash
python3.11 - <<'PY'
import getpass
import os
import turbopuffer as tpuf

namespace = os.environ["TURBOPUFFER_NAMESPACE"]
api_key = getpass.getpass("TurboPuffer API key: ")
client = tpuf.Turbopuffer(
    api_key=api_key,
    base_url=os.environ.get("TURBOPUFFER_API_BASE_URL", "https://api.turbopuffer.com"),
)
client.namespace(namespace).delete_all()
print(f"deleted namespace: {namespace}")
PY
```

For a normal fresh run, `--drop-old --load` also deletes the configured
namespace at run start. The explicit teardown is useful when verifying a clean
SaaS state before launching the benchmark.

## Validated Baseline

Run details:

- Date: 2026-05-28
- Client: `/home/ubuntu/VectorDBBench` on branch `fts`
- TurboPuffer SDK: `2.0.0`
- API base URL: `https://api.turbopuffer.com`
- Namespace: `vdbbench-fts-tpuf-20260528-1202-msmarco-small`
- Full load run ID: `f19622098a6f460c93064446e8700726`
- Search-only run ID: `7417b907e9654818a913e27b3d68e093`
- Result file: `/tmp/vectordb_bench/results/TurboPuffer/result_20260528_fts-e2e-tpuf-msmarco-small-searchonly_turbopuffer.json`
- Status: normal

Namespace state after load:

- Rows: `100000`
- Approx logical bytes: `32513593`
- Index status: `up-to-date`
- Last write: `2026-05-28T12:22:17Z`

Load metrics from the full ingest run:

- Insert duration: `1111.388s`
- Optimize/warmup duration: `60.034s`
- Load duration: `1171.4219s`

Search metrics from the successful search-only run:

- Best QPS: `146.5862`
- Recall: `0.9000`
- NDCG: `0.6860`
- MRR: `0.6322`
- Serial average latency: `0.1337s`
- Serial p99 latency: `0.2280s`
- Serial p95 latency: `0.1869s`
- Concurrent QPS by concurrency `[1, 5, 10, 20]`: `[8.2367, 35.5999, 72.4753, 146.5862]`
- Concurrent p99 latency: `[0.1834461819, 0.2421539293, 0.2269624966, 0.2226006012]`
- Concurrent p95 latency: `[0.1753568540, 0.1960341744, 0.1942978568, 0.1987305184]`
- Concurrent average latency: `[0.1214004508, 0.1401809676, 0.1376166602, 0.1359992910]`

Baseline caveats:

- This is a SaaS backend result, so it is affected by TurboPuffer service-side
  conditions outside the EC2 hosts.
- The final JSON is from a search-only rerun because the initial full run
  completed ingest and warmup but hit the SDK-client pickling issue at search.
- The client host was not isolated from other session work. Treat this as a
  functional/comparable first pass, not a clean isolated performance number.
