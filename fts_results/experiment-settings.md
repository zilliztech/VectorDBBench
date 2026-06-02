# Current FTS E2E Experiment Setting

Generated on 2026-06-02.

## Scope

The draft-1 FTS benchmark matrix records MS MARCO and HotpotQA runs across Milvus, Elasticsearch, Vespa, and TurboPuffer. For the larger local-server rebench, TurboPuffer is skipped and the target larger cases are the Medium datasets:

- `MS MARCO Medium (1M documents)`
- `HotpotQA Medium (1M documents)`

Large datasets remain out of scope for draft 1 unless the server resources are increased and a separate large-run plan is approved.

## Two-Machine Setup

Client role:

- AWS EC2 `i8g.2xlarge`
- Ubuntu 22.04, Linux 6.8, `aarch64`
- 8 vCPU, Neoverse-V2, 1 thread per core
- About 61 GiB RAM
- 485 GiB root filesystem, shared by the repo, temporary files, and dataset cache
- Runs VectorDBBench from `/home/ubuntu/VectorDBBench`
- Uses `DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset`
- Uses Python `3.11`

Server role:

- AWS EC2 `m5d.2xlarge`
- Amazon Linux 2023, Linux 6.1, `x86_64`
- 8 vCPU, Intel Xeon Platinum 8175M, 2 threads per core
- About 30 GiB usable RAM plus 31 GiB swap
- 500 GiB root filesystem
- 279 GiB local NVMe mounted at `/data`
- Runs one backend deployment at a time via Docker

Exact host addresses, SSH material, and tokens are intentionally excluded from this repository.

## Backend Deployment Baselines

Milvus:

- Follow `docs/fts-backends/milvus.md`
- Standalone Docker Compose deployment
- Milvus image/version from the backend runbook
- Fresh teardown before each backend/dataset run

Elasticsearch:

- Follow `docs/fts-backends/elasticsearch.md`
- Image `docker.elastic.co/elasticsearch/elasticsearch:8.16.0`
- Docker memory limit `-m 8g`
- No explicit `ES_JAVA_OPTS` in the current runbook unless a later run bundle records otherwise
- Fresh Docker volume before each backend/dataset run

Vespa:

- Follow `docs/fts-backends/vespa.md`
- Image `vespaengine/vespa:8.694.53`
- Data and logs under `/srv/vespa`
- Fresh `/srv/vespa/var` and `/srv/vespa/logs` before each backend/dataset run

TurboPuffer:

- Result directories are reserved for prior and future TurboPuffer FTS runs
- Tokens and namespace secrets must not be stored here
- Skipped for the larger local-server rebench

## Run Policy

- Run from the client machine against the server machine.
- Run only one backend at a time on the server.
- Tear down and redeploy the backend before every backend/dataset benchmark.
- Use the backend-specific runbook exactly unless the run bundle records an explicit deviation.
- Preserve raw result JSON and logs for every completed run.
- Record server-side caveats such as concurrent build processes, container restarts, disk pressure, or JVM/container memory tuning.
