# Turbopuffer MS MARCO FTS E2E Report

## Prequsites

- Backend: TurboPuffer managed service.
- Dataset family: MS MARCO.
- Current committed raw results: `MS MARCO Small (100K documents)`.
- Source runbook: `docs/fts-backends/turbopuffer.md`.
- Raw result directory: `raw_results/`.
- Current result JSONs have API key fields masked by VectorDBBench.

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
- Region for committed full run: `aws-us-east-1`.
- Namespace for committed full run: `vdbbench-fts-tpuf-20260601-1042-msmarco-small`.
- TurboPuffer service-side CPU, memory, disk, and replica placement are managed externally and are not visible from the benchmark host.

## Server Setup

TurboPuffer is managed, so server setup is namespace and client credential setup rather than Docker deployment. Do not commit the token or generated config file.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export TURBOPUFFER_REGION="aws-us-east-1"
export TURBOPUFFER_NAMESPACE="vdbbench-fts-tpuf-$(date -u +%Y%m%d-%H%M)-msmarco-small"

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

Exact client script for the committed full MS MARCO Small run:

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
  --task-label "fts-e2e-tpuf-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The 2026-05-28 JSON is a search-only recovery result from the first run and has `load_duration=0.0`.

Effective TurboPuffer FTS case config from the full 2026-06-01 raw JSON:

- `region=aws-us-east-1`
- `time_wait_warmup=60`
- `pin_namespace=false`
- `pin_namespace_requested=false`
- `pin_replicas=1`
- `pin_timeout=2700`
- `scalar_payload_label_field=label`

## Result

| Raw JSON | Task label | Dataset size | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/5/10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260528_fts-e2e-tpuf-msmarco-small-searchonly_turbopuffer.json` | `fts-e2e-tpuf-msmarco-small-searchonly` | 100K | 0.0000 | 146.5862 | 0.9000 | 0.6860 | 0.6322 | 0.1869 | 0.2280 | 8.2367 / 35.5999 / 72.4753 / 146.5862 |
| `result_20260601_fts-e2e-tpuf-msmarco-small_turbopuffer.json` | `fts-e2e-tpuf-msmarco-small` | 100K | 290.5625 | 257.3771 | 0.9125 | 0.7156 | 0.6659 | 0.0840 | 0.1081 | 1.3357 / 49.4967 / 126.8548 / 257.3771 |
