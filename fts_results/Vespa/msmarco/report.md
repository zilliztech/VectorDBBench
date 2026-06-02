# Vespa MS MARCO FTS E2E Report

## Prequsites

- Backend: Vespa single-node container.
- Dataset family: MS MARCO.
- Current committed raw results: `MS MARCO Small (100K documents)` and `MS MARCO Medium (1M documents)` on the original `m5d.2xlarge` server and the later `r7i.4xlarge` server.
- Run dates represented here: 2026-05-28, 2026-06-01, and 2026-06-02.
- Source runbook: `docs/fts-backends/vespa.md`.
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

Original server machine:

- EC2 type: `m5d.2xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 8 vCPU, Intel Xeon Platinum 8175M, 4 physical cores, 2 threads per core.
- Memory: about 30 GiB usable RAM plus 31 GiB swap.
- Disk quota: root XFS filesystem 500 GiB total with 451 GiB available at last check; additional 279 GiB NVMe mounted at `/data`.
- Docker: overlay2 under `/var/lib/docker` on the root filesystem.
- Role: runs the Vespa container and `/srv/vespa` state directories.

Rerun server machine:

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Vespa container and `/srv/vespa` state directories for the `r7i` rerun.

## Server Setup

Validated deployment:

- Vespa image: `vespaengine/vespa:8.694.53`.
- Container name: `vespa`.
- Hostname: `vespa-container`.
- Persistent host directories: `/srv/vespa/var` and `/srv/vespa/logs`.
- Ports: `8080` for feed/query and `19071` for config/deploy.

Reproducible fresh-deploy script:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo mkdir -p /srv/vespa/var /srv/vespa/logs
sudo chown -R 1000:1000 /srv/vespa/var /srv/vespa/logs

sudo docker rm -f vespa >/dev/null 2>&1 || true
sudo find /srv/vespa/var -mindepth 1 -delete
sudo find /srv/vespa/logs -mindepth 1 -delete
sudo docker pull vespaengine/vespa:8.694.53

sudo docker run -d --name vespa --user vespa:vespa --hostname vespa-container \
  --ulimit nofile=262144:262144 --pids-limit=-1 \
  -v /srv/vespa/var:/opt/vespa/var \
  -v /srv/vespa/logs:/opt/vespa/logs \
  -p 0.0.0.0:8080:8080 \
  -p 0.0.0.0:19071:19071 \
  --restart unless-stopped \
  vespaengine/vespa:8.694.53

curl -fsS http://127.0.0.1:19071/state/v1/health
```

Fresh teardown script used after the `r7i` rerun:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f vespa >/dev/null 2>&1 || true
sudo find /srv/vespa/var -mindepth 1 -delete
sudo find /srv/vespa/logs -mindepth 1 -delete
sudo docker ps -a
sudo docker volume ls
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

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "http://${SERVER_HOST}" \
  --port "8080" \
  --task-label "fts-e2e-vespa-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The `r7i.4xlarge` rerun used the same command with task label `fts-e2e-vespa-msmarco-small-r7i`.
The `r7i.4xlarge` medium run used the same command with task label `fts-e2e-vespa-msmarco-medium-r7i` and dataset size `MS MARCO Medium (1M documents)`.

Effective Vespa FTS case config from the raw JSON: no backend-specific case fields are set. The VDBBench Vespa adapter deploys the application package through port `19071` and queries through port `8080`.

## Result

| Raw JSON | Task label | Dataset size | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/5/10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `result_20260528_fts-e2e-vespa-msmarco-small_vespa.json` | `fts-e2e-vespa-msmarco-small` | 100K | 171.9693 | 478.5384 | 0.9416 | 0.7509 | 0.7015 | 0.0212 | 0.0269 | 63.4393 / 146.3972 / 428.5583 / 478.5384 |
| `result_20260601_fts-e2e-vespa-msmarco-small_vespa.json` | `fts-e2e-vespa-msmarco-small` | 100K | 81.6402 | 482.1899 | 0.9416 | 0.7509 | 0.7015 | 0.0202 | 0.0260 | 78.8898 / 336.8178 / 482.1899 / 470.0384 |
| `result_20260602_fts-e2e-vespa-msmarco-small-r7i_vespa.json` | `fts-e2e-vespa-msmarco-small-r7i` | 100K | 79.2473 | 734.5241 | 0.9416 | 0.7509 | 0.7015 | 0.0184 | 0.0230 | 91.7244 / 512.5352 / 347.9730 / 734.5241 |
| `result_20260602_fts-e2e-vespa-msmarco-medium-r7i_vespa.json` | `fts-e2e-vespa-msmarco-medium-r7i` | 1M | 585.7496 | 196.8213 | 0.8409 | 0.5499 | 0.4767 | 0.1268 | 0.1744 | 17.3137 / 85.2530 / 139.2205 / 196.8213 |

Latest `r7i.4xlarge` rerun vs previous `m5d.2xlarge` run:

- QPS increased from `482.1899` to `734.5241` (+52.3%).
- Load duration changed from `81.6402s` to `79.2473s` (-2.9%).
- Recall stayed unchanged at `0.9416`.
- p95 changed from `0.0202s` to `0.0184s`; p99 changed from `0.0260s` to `0.0230s`.
