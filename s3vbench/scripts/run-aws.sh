#!/usr/bin/env sh
set -eu

if [ "$#" -lt 4 ]; then
  cat >&2 <<'USAGE'
usage:
  scripts/run-aws.sh <endpoint> <access-key> <secret-key> <command> [s3vbench flags...]

example:
  scripts/run-aws.sh https://s3vectors.us-east-1.amazonaws.com AKIA... SECRET... put \
    --region us-east-1 \
    --vector-bucket my-vector-bucket \
    --index my-index \
    --dimension 768 \
    --requests 10000 \
    --concurrency 256 \
    --put-batch-size 500

Optional:
  BIN=dist/s3vbench scripts/run-aws.sh ...
  AWS_SESSION_TOKEN=... scripts/run-aws.sh ...
USAGE
  exit 2
fi

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${BIN:-"$ROOT_DIR/dist/s3vbench"}
ENDPOINT=$1
ACCESS_KEY=$2
SECRET_KEY=$3
COMMAND=$4
shift 4

if [ ! -x "$BIN" ]; then
  echo "binary not found or not executable: $BIN" >&2
  echo "build first: TARGETS=$(go env GOOS)/$(go env GOARCH) scripts/build.sh" >&2
  exit 2
fi

exec "$BIN" "$COMMAND" \
  --client aws \
  --endpoint "$ENDPOINT" \
  --access-key "$ACCESS_KEY" \
  --secret-key "$SECRET_KEY" \
  --session-token "${AWS_SESSION_TOKEN:-}" \
  "$@"
