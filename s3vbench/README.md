# s3vbench

Dedicated benchmark runner for Amazon S3 Vectors workloads.

This Phase 1 implementation focuses on the locally testable benchmark core:

- operation commands: `put`, `query`, `query-filter`, `get`, `list`, `delete`, `mixed`
- deterministic synthetic vector generation
- count-based and duration-based execution
- configurable concurrency and optional target QPS
- mixed workload ratios by request count
- aggregate and per-operation latency/QPS/error metrics
- report artifacts: `summary.json`, `timeseries.csv`, `timeseries.ndjson`, and `errors.jsonl` when errors occur

The current executable uses the built-in `mock` client for local validation. The runner is built around a `Client` interface so the AWS SDK for Go v2 `service/s3vectors` adapter can be added without changing the engine, report, or CLI contracts.

## Examples

```bash
GO111MODULE=on go run ./cmd/s3vbench put \
  --vector-bucket my-vector-bucket \
  --index my-index \
  --dimension 768 \
  --requests 10000 \
  --batch-size 500 \
  --concurrency 256
```

```bash
GO111MODULE=on go run ./cmd/s3vbench mixed \
  --vector-bucket my-vector-bucket \
  --index my-index \
  --dimension 768 \
  --duration 30s \
  --concurrency 512 \
  --mix query=90,put=5,get=3,delete=1,list=1
```

```bash
GO111MODULE=on go run ./cmd/s3vbench query-filter \
  --vector-bucket my-vector-bucket \
  --index my-index \
  --dimension 768 \
  --requests 1000 \
  --filter-file filters.jsonl
```

`filters.jsonl` contains one scalar/metadata filter object per line.

## Test

```bash
GO111MODULE=on GOCACHE=/path/to/writable/cache go test ./...
```

In this workspace, the global Go environment has `GO111MODULE=off`, so project commands should set `GO111MODULE=on`.

## Build

```bash
make test
make build
```

Cross-compile common Linux and macOS binaries:

```bash
scripts/build.sh
```

Limit targets when needed:

```bash
TARGETS="linux/amd64 linux/arm64" scripts/build.sh
```

Run locally through the project wrapper:

```bash
scripts/run-local.sh mixed \
  --vector-bucket my-vector-bucket \
  --index my-index \
  --dimension 768 \
  --requests 1000 \
  --mix query=90,put=5,get=3,delete=1,list=1
```

Run against an S3 Vectors endpoint after building:

```bash
TARGETS="$(go env GOOS)/$(go env GOARCH)" scripts/build.sh

scripts/run-aws.sh https://s3vectors.us-east-1.amazonaws.com AKIA... SECRET... put \
  --region us-east-1 \
  --vector-bucket my-vector-bucket \
  --index my-index \
  --dimension 768 \
  --requests 10000 \
  --concurrency 256 \
  --put-batch-size 500
```

For temporary credentials, set `AWS_SESSION_TOKEN` before invoking `scripts/run-aws.sh`.
