# FTS BM25 End-to-End Testing Plan

This plan validates the first-pass BM25 full-text-search implementation across
the supported backend and dataset matrix.

## Scope

Backends:

- Milvus
- Elasticsearch through the existing `elasticcloudhnsw` backend command
- Vespa
- TurboPuffer

Default datasets:

- MS MARCO Small (100K documents)
- MS MARCO Medium (1M documents)
- HotpotQA Small (100K documents)
- HotpotQA Medium (1M documents)

## Backend Runbooks

Keep this document as the shared FTS BM25 matrix and common configuration
reference. Put backend-specific deployment, health checks, benchmark commands,
monitoring, result validation, and baseline observations in one document per
backend:

- Milvus: `docs/fts-backends/milvus.md`
- Elasticsearch: `docs/fts-backends/elasticsearch.md`
- Vespa: `docs/fts-backends/vespa.md`
- TurboPuffer: `docs/fts-backends/turbopuffer.md` (TODO)

Out of scope for the first E2E pass:

- Large datasets:
  - MS MARCO Large (8.8M documents)
  - HotpotQA Large (5.2M documents)
- Payload/text retrieval.
- ClickHouse token/boolean FTS.
- Hybrid dense+sparse retrieval.

## Service Availability Notes

Elastic:

- Elastic Cloud Hosted is a paid managed service with a free 14-day trial. The
  official pricing page describes hosted pricing as resource based and
  pay-as-you-go.
- Elastic self-managed has a "Free and open" tier. This is a viable no-service
  option for Elasticsearch itself.
- The VDBBench backend enum and command are still named `ElasticCloud` /
  `elasticcloudhnsw` for compatibility, but the connection config now supports
  both managed Elastic Cloud (`--cloud-id`) and self-hosted Elasticsearch
  (`--host`, `--port`).
- For this E2E plan, use self-hosted Elasticsearch by default. Elastic Cloud is
  an alternate run mode for teams that already have hosted credentials.

References:

- https://www.elastic.co/pricing
- https://www.elastic.co/pricing/faq
- https://www.elastic.co/pricing/self-managed
- https://www.elastic.co/elasticsearch/service
- https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-docker-basic
- https://www.elastic.co/docs/reference/elasticsearch/clients/python/connecting
- https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query
- https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity

Vespa:

- Vespa is available as open source and can be self-hosted.
- Vespa Cloud is a managed service with pricing by allocated resources. The
  free trial includes usage credits and stops the application when credits run
  out rather than billing beyond the credits.
- The current VDBBench `vespa` command takes `--uri` and `--port`, so the
  recommended first E2E target is a self-hosted/local Vespa endpoint.

References:

- https://blog.vespa.ai/open-sourcing-vespa-yahoos-big-data-processing/
- https://vespa.ai/free-trial/
- https://cloud.vespa.ai/price-calculator
- https://cloud.vespa.ai/
- https://docs.vespa.ai/en/basics/deploy-an-application-local.html
- https://docs.vespa.ai/en/operations/self-managed/docker-containers.html
- https://docs.vespa.ai/en/ranking/bm25.html

Milvus:

- Milvus standalone is available as open source and can be self-hosted with
  Docker Compose.
- Milvus BM25 full-text search requires Milvus 2.5.0 or newer. For the first
  reproducible E2E baseline, pin the standalone image version instead of using
  an unversioned or moving `latest` target.

References:

- https://milvus.io/docs/install_standalone-docker-compose.md
- https://milvus.io/docs/full-text-search.md
- https://milvus.io/docs/analyzer-overview.md

## Common Environment

Set these before running the matrix:

```bash
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100

export MILVUS_URI="http://..."
export MILVUS_USER=""
export MILVUS_PASSWORD=""

export ELASTIC_HOST="127.0.0.1"
export ELASTIC_PORT="9200"
export ELASTIC_USER=""
export ELASTIC_CLOUD_ID="..."
export ELASTIC_PASSWORD="..."

export VESPA_URI="http://127.0.0.1"
export VESPA_PORT="8080"

export TURBOPUFFER_API_KEY="..."
export TURBOPUFFER_API_BASE_URL="https://api.turbopuffer.com"
export TURBOPUFFER_NAMESPACE_PREFIX="vdbbench-fts-e2e"
```

Use `python3.11 -m vectordb_bench.cli.vectordbbench` from the repository root.
Do not use bare `pytest` or an unpinned Python executable for local validation.

FTS datasets are loaded through `ir_datasets` regardless of the global
`DATASET_SOURCE` setting. By default, `ir_datasets` stores downloaded files
under `$DATASET_LOCAL_DIR/ir_datasets` and temporary files under
`$DATASET_LOCAL_DIR/ir_datasets_tmp`.

## Common Benchmark Options

Each E2E run should include:

```bash
--case-type FTSmsmarcoPerformance
--drop-old
--load
--search-serial
--search-concurrent
--k 100
--concurrency-duration 30
--num-concurrency "1,5,10,20"
--concurrency-timeout 3600
```

The default VDBBench concurrency list is larger (`1,5,10,20,30,40,60,80`).
For the first functional E2E pass, `1,5,10,20` is enough to exercise concurrent
search without turning validation into a full leaderboard run.

## Standalone Backend Deployment

These steps assume the database server and VDBBench client may be different
machines. Run the server deployment commands on the backend host. Run all
`python3.11 -m vectordb_bench.cli.vectordbbench ...` commands from the
repository root on the VDBBench client.

### Milvus Standalone

Deploy Milvus standalone with Docker Compose. Pin the version used for a test
run; the first recommended baseline is `v2.6.17`.

```bash
mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

export MILVUS_VERSION=v2.6.17
wget "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml" \
  -O docker-compose.yml

docker compose pull
docker compose up -d
docker compose ps
```

Milvus standalone uses etcd and MinIO internally. The VDBBench client only needs
Milvus proxy port `19530`. Keep etcd and MinIO private. Port `9091` is useful
for private health checks but should not be exposed publicly.

Server-side health check:

```bash
cd ~/milvus-standalone
docker compose ps
curl -fsS http://127.0.0.1:9091/healthz
docker logs --tail 100 milvus-standalone
```

Client-side connectivity check:

```bash
export SERVER_IP="<milvus-server-ip-or-dns>"
export MILVUS_URI="http://${SERVER_IP}:19530"

nc -vz "$SERVER_IP" 19530

python3.11 - <<'PY'
import os
from pymilvus import connections, utility

connections.connect(
    uri=os.environ["MILVUS_URI"],
    user=os.environ.get("MILVUS_USER") or None,
    password=os.environ.get("MILVUS_PASSWORD") or None,
    timeout=10,
)
print("server_version:", utility.get_server_version())
print("collections:", utility.list_collections())
PY
```

If auth is enabled, set `MILVUS_USER` and `MILVUS_PASSWORD`; otherwise leave
them empty. The current VDBBench `milvusfts` command does not expose TLS
certificate options, so use plaintext only on a private network, VPN, or SSH
tunnel unless TLS support is added later.

Milvus FTS caveats:

- VDBBench creates `doc_id`, `text`, and `sparse_vector` fields.
- The `text` field must be analyzer-enabled `VARCHAR`.
- A `FunctionType.BM25` function generates sparse vectors from text.
- The sparse vector field is indexed with BM25 using `SPARSE_INVERTED_INDEX`.
- Analyzer behavior affects both result quality and ingest/search latency.

### Elasticsearch Standalone

Deploy a single-node Elasticsearch server. Pin the version used for a test run;
the first recommended baseline is `8.16.0`, matching the current Python client
dependency in this branch.

For an isolated benchmark-only network, the simplest deployment disables
security. Do not expose this mode publicly.

```bash
export ES_VERSION=8.16.0

sudo sysctl -w vm.max_map_count=1048576
docker volume create esdata01

docker run -d --name es01 \
  -p 0.0.0.0:9200:9200 \
  --restart unless-stopped \
  --ulimit nofile=65535:65535 \
  -m 8g \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -v esdata01:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}
```

For a reachable shared server, prefer Elastic's secure Docker defaults and use a
password. Current VDBBench supports username/password and TLS verification
toggling, but not CA certificate or fingerprint flags. For local secure Docker
with the generated private CA, use `--use-ssl --no-verify-certs` only for E2E
benchmark validation.

```bash
export ES_VERSION=8.16.0
export ELASTIC_PASSWORD="replace-with-strong-password"

sudo sysctl -w vm.max_map_count=1048576
docker network create elastic || true
docker volume create esdata01

printf '%s' "$ELASTIC_PASSWORD" > bootstrapPassword.txt
chmod 0444 bootstrapPassword.txt

docker run -d --name es01 --net elastic \
  -p 0.0.0.0:9200:9200 \
  --restart unless-stopped \
  --ulimit nofile=65535:65535 \
  -m 8g \
  -e "discovery.type=single-node" \
  -e ELASTIC_PASSWORD_FILE=/run/secrets/bootstrapPassword.txt \
  -v "$PWD/bootstrapPassword.txt:/run/secrets/bootstrapPassword.txt:ro" \
  -v esdata01:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}
```

The VDBBench client only needs Elasticsearch HTTP REST port `9200`. Port `9300`
is the node transport port and is not needed for this single-node client
benchmark.

Health checks:

```bash
export ELASTIC_HOST="<elasticsearch-server-ip-or-dns>"
export ELASTIC_PORT=9200

curl "http://${ELASTIC_HOST}:${ELASTIC_PORT}/"
curl "http://${ELASTIC_HOST}:${ELASTIC_PORT}/_cluster/health?pretty&wait_for_status=yellow&timeout=60s"
curl "http://${ELASTIC_HOST}:${ELASTIC_PORT}/_cat/nodes?v"
```

For secure Docker:

```bash
export ELASTIC_USER=elastic
export ELASTIC_PASSWORD="replace-with-strong-password"

curl -k -u "${ELASTIC_USER}:${ELASTIC_PASSWORD}" "https://${ELASTIC_HOST}:${ELASTIC_PORT}/"
curl -k -u "${ELASTIC_USER}:${ELASTIC_PASSWORD}" \
  "https://${ELASTIC_HOST}:${ELASTIC_PORT}/_cluster/health?pretty&wait_for_status=yellow&timeout=60s"
```

Elasticsearch FTS caveats:

- VDBBench maps `doc_id` as `keyword` and `text` as `text`.
- Search uses a plain `match` query on the `text` field.
- Elasticsearch BM25 is the default text similarity.
- No analyzer, stemming, stopword, synonym, phrase, fuzzy, or hybrid options are
  exposed in this first draft.
- Use one shard and zero replicas for the first baseline to reduce tie-order and
  shard-layout variance.

### Vespa Standalone

Deploy Vespa as a single Docker container. The VDBBench client deploys the Vespa
application package during the run, so both the config/deploy API and the
query/feed API must be reachable.

Disposable deployment:

```bash
docker rm -f vespa || true
docker run -d --name vespa --hostname vespa-container \
  -p 8080:8080 -p 19071:19071 \
  vespaengine/vespa:8.694.53
```

Persistent deployment:

```bash
sudo mkdir -p /srv/vespa/{var,logs}
sudo chown -R 1000:1000 /srv/vespa/{var,logs}

docker rm -f vespa || true
docker run -d --name vespa --user vespa:vespa --hostname vespa-container \
  --ulimit nofile=262144:262144 --pids-limit=-1 \
  -v /srv/vespa/var:/opt/vespa/var \
  -v /srv/vespa/logs:/opt/vespa/logs \
  -p 8080:8080 -p 19071:19071 \
  vespaengine/vespa:8.694.53
```

The VDBBench client needs:

- `19071` for Vespa deploy/config API.
- `8080` for feed/query API.

Set `VESPA_URI` to the base URI without a port; the current VDBBench adapter
uses `VESPA_URI:19071` internally for deployment and `VESPA_URI:VESPA_PORT` for
feed/query.

```bash
export VESPA_URI="http://<vespa-server-ip-or-dns>"
export VESPA_PORT=8080
```

Do not set `VESPA_URI=http://host:8080`.

Health checks:

```bash
curl -fsS "$VESPA_URI:19071/state/v1/health"
curl -fsS "$VESPA_URI:$VESPA_PORT/state/v1/health"
curl -fsS "$VESPA_URI:$VESPA_PORT/status.html"
docker logs --tail=200 vespa
```

Self-hosted Vespa defaults to unauthenticated HTTP. The current VDBBench Vespa
CLI has no certificate, key, CA, token, or auth flags, so use a private network,
VPN, firewall, or SSH tunnel for remote testing:

```bash
ssh -N -L 8080:127.0.0.1:8080 -L 19071:127.0.0.1:19071 user@vespa-server
export VESPA_URI=http://127.0.0.1
export VESPA_PORT=8080
```

Vespa FTS caveats:

- VDBBench deploys the Vespa app automatically at run start.
- The FTS app uses a `text` field with `index: enable-bm25`.
- The rank profile uses `bm25(text)`.
- Queries use `userQuery()`, `type=any`, `ranking=bm25`, `hits=k`, and
  `default-index=text`.
- BM25/analyzer knobs are not exposed by VDBBench's first-draft Vespa FTS CLI.

### TurboPuffer Hosted

TurboPuffer is hosted-only for this plan. There is no standalone server
deployment step. Set API credentials on the VDBBench client and run the
TurboPuffer matrix below.

## Preflight

Run the focused FTS unit/integration suite:

```bash
python3.11 -m pytest \
  tests/test_fts_dataset.py \
  tests/test_fts_cases.py \
  tests/test_fts_runners.py \
  tests/test_fts_backend_capability.py \
  tests/test_fts_milvus.py \
  tests/test_fts_elastic_cloud.py \
  tests/test_fts_vespa.py \
  tests/test_fts_turbopuffer.py \
  tests/test_fts_format_results.py \
  -q
```

Expected result:

- All focused FTS tests pass.
- Current known local warning from `pytz` is acceptable.

Run one dry-run per backend to verify CLI argument parsing and case config
selection:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --dry-run \
  --uri "$MILVUS_URI" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --dry-run \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --dry-run \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --dry-run \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-dryrun" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)"
```

Note: `elasticcloudhnsw`, `vespa`, and `turbopuffer` are existing backend CLI
commands. When `--case-type FTSmsmarcoPerformance` is selected, the shared CLI
logic swaps their DB case config to the backend FTS config.

## Milvus Matrix

```bash
python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "$MILVUS_URI" \
  --user-name "$MILVUS_USER" \
  --password "$MILVUS_PASSWORD" \
  --task-label "fts-e2e-milvus-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## ElasticCloud Matrix

The command name remains `elasticcloudhnsw`, but these runs target self-hosted
Elasticsearch. If your local cluster requires authentication, add
`--user-name "$ELASTIC_USER" --password "$ELASTIC_PASSWORD"` to each command. If
the cluster uses HTTPS, add `--use-ssl`; if it uses a private CA or self-signed
certificate for HTTPS during local testing, add `--no-verify-certs`.

```bash
python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --task-label "fts-e2e-elastic-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --task-label "fts-e2e-elastic-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --task-label "fts-e2e-elastic-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "$ELASTIC_HOST" \
  --port "$ELASTIC_PORT" \
  --task-label "fts-e2e-elastic-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

Elastic Cloud alternate:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --cloud-id "$ELASTIC_CLOUD_ID" \
  --password "$ELASTIC_PASSWORD" \
  --task-label "fts-e2e-elastic-cloud-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## Vespa Matrix

```bash
python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench vespa \
  --uri "$VESPA_URI" \
  --port "$VESPA_PORT" \
  --task-label "fts-e2e-vespa-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## TurboPuffer Matrix

Use a separate namespace for each run so cleanup is isolated:

```bash
python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-msmarco-small" \
  --task-label "fts-e2e-tpuf-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-msmarco-medium" \
  --task-label "fts-e2e-tpuf-msmarco-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-hotpotqa-small" \
  --task-label "fts-e2e-tpuf-hotpotqa-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --api-base-url "$TURBOPUFFER_API_BASE_URL" \
  --namespace "${TURBOPUFFER_NAMESPACE_PREFIX}-hotpotqa-medium" \
  --task-label "fts-e2e-tpuf-hotpotqa-medium" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 --num-concurrency "1,5,10,20" --concurrency-timeout 3600
```

## Validation After Each Run

For each task label:

1. Confirm the command exits with status `0`.
2. Confirm a result JSON is written under `$RESULTS_LOCAL_DIR/<DB>/`.
3. Confirm the result JSON is readable by `TestResult.read_file`.
4. Confirm the performance metrics include:
   - `insert_duration`
   - `optimize_duration`
   - `load_duration`
   - `recall`
   - `ndcg`
   - `mrr`
   - `serial_latency_p99`
   - `serial_latency_p95`
   - `qps`
   - `conc_qps_list`
   - `conc_latency_p99_list`
   - `conc_latency_p95_list`
5. Confirm logs do not contain:
   - `NotImplementedError`
   - wrong DB case config type errors
   - dataset/qrel validation errors
   - systemic empty search result failures
   - insert retry exhaustion

Result directory names use the VDBBench DB enum values:

- `$RESULTS_LOCAL_DIR/Milvus/`
- `$RESULTS_LOCAL_DIR/ElasticCloud/`
- `$RESULTS_LOCAL_DIR/Vespa/`
- `$RESULTS_LOCAL_DIR/TurboPuffer/`

Result files use:

```text
result_YYYYMMDD_<task-label>_<db>.json
```

For example:

```text
$RESULTS_LOCAL_DIR/Milvus/result_20260528_fts-e2e-milvus-msmarco-small_milvus.json
```

Using the same task label on the same date overwrites the previous result file.

Validate a result JSON with the project parser:

```bash
export RESULT_PATH="$RESULTS_LOCAL_DIR/Milvus/result_20260528_fts-e2e-milvus-msmarco-small_milvus.json"

python3.11 - <<'PY'
import os
from pathlib import Path

from vectordb_bench.models import TestResult

result = TestResult.read_file(Path(os.environ["RESULT_PATH"]))
assert len(result.results) >= 1
case_result = result.results[0]
assert case_result.label.name == "NORMAL"
assert case_result.task_config.case_config.case_id.name == "FTSmsmarcoPerformance"
assert case_result.metrics.recall >= 0
assert case_result.metrics.ndcg >= 0
assert case_result.metrics.mrr >= 0
print(result.run_id, result.task_label, case_result.metrics)
PY
```

## Stop Conditions

Stop the matrix and debug before continuing if any backend has:

- insert failure after retries
- optimize timeout
- serial search failure
- all-zero `recall`, `ndcg`, and `mrr` for a completed run
- missing or unreadable result JSON
- backend API/schema error indicating adapter incompatibility

## Optional Phase: Large Datasets

Run only after all default dataset runs pass:

- MS MARCO Large (8.8M documents)
- HotpotQA Large (5.2M documents)

Use the same backend commands and replace `--dataset-with-size-type` with the
Large tier. Treat this as capacity/performance validation, not first-pass
functional validation.
