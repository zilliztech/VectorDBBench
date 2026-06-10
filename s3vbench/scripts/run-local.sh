#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GOCACHE=${GOCACHE:-"$ROOT_DIR/.cache/go-build"}

cd "$ROOT_DIR"
GO111MODULE=on GOCACHE="$GOCACHE" go run ./cmd/s3vbench "$@"
