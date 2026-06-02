# Vespa HotpotQA FTS E2E Report

## Prequsites

- Backend: Vespa single-node container.
- Dataset family: HotpotQA.
- Current committed raw results: none.
- Source runbook: `docs/fts-backends/vespa.md`.
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
- Role: runs the Vespa container and `/srv/vespa` state directories.

## Server Setup

Use the same Vespa setup as `../msmarco/report.md`: image `vespaengine/vespa:8.694.53`, container `vespa`, `/srv/vespa/var`, `/srv/vespa/logs`, and a fresh directory cleanup before each run.

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo mkdir -p /srv/vespa/var /srv/vespa/logs
sudo chown -R 1000:1000 /srv/vespa/var /srv/vespa/logs

docker rm -f vespa >/dev/null 2>&1 || true
sudo find /srv/vespa/var -mindepth 1 -delete
sudo find /srv/vespa/logs -mindepth 1 -delete
docker pull vespaengine/vespa:8.694.53

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

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "http://${SERVER_HOST}" \
  --port "8080" \
  --task-label "fts-e2e-vespa-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

Expected Vespa FTS case config: no backend-specific case fields are set. The VDBBench Vespa adapter deploys the application package through port `19071` and queries through port `8080`.

## Result

No HotpotQA Vespa raw result JSON has been committed yet.
