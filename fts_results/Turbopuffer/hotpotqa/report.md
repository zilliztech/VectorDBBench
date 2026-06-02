# Turbopuffer HotpotQA FTS E2E Report

## Prequsites

- Backend: TurboPuffer managed service.
- Dataset family: HotpotQA.
- Current committed raw results: none.
- Source runbook: `docs/fts-backends/turbopuffer.md`.
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

Server/service side:

- No benchmark server-hosting EC2 deployment is used for TurboPuffer.
- TurboPuffer service-side CPU, memory, disk, and replica placement are managed externally and are not visible from the benchmark host.

## Server Setup

TurboPuffer is managed, so server setup is namespace and client credential setup rather than Docker deployment. Do not commit the token or generated config file.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export TURBOPUFFER_REGION="aws-us-east-1"
export TURBOPUFFER_NAMESPACE="vdbbench-fts-tpuf-$(date -u +%Y%m%d-%H%M)-hotpotqa-medium"

TPUF_CONFIG="$(mktemp /tmp/turbopuffer-vdbbench.XXXXXX.yml)"
chmod 600 "$TPUF_CONFIG"

python3.11 - <<'PY' "$TPUF_CONFIG" "$TURBOPUFFER_NAMESPACE" "$TURBOPUFFER_REGION"
import getpass
import sys

path, namespace, region = sys.argv[1:4]
api_key = getpass.getpass("TurboPuffer API key: ")
with open(path, "w", encoding="utf-8") as f:
    f.write("turbopuffer:\n")
    f.write(f"  api_key: {api_key}\n")
    f.write(f"  region: {region}\n")
    f.write(f"  namespace: {namespace}\n")
PY

printf 'TPUF_CONFIG=%s\n' "$TPUF_CONFIG"
printf 'TURBOPUFFER_NAMESPACE=%s\n' "$TURBOPUFFER_NAMESPACE"
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
export TPUF_CONFIG="<path-to-local-secret-config-yml>"

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --config-file "$TPUF_CONFIG" \
  --task-label "fts-e2e-tpuf-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected TurboPuffer FTS case config:

- `region=aws-us-east-1`
- `time_wait_warmup=60`
- `pin_namespace=false`
- `pin_namespace_requested=false`
- `pin_replicas=1`
- `pin_timeout=2700`
- `scalar_payload_label_field=label`

## Result

No HotpotQA TurboPuffer raw result JSON has been committed yet.
