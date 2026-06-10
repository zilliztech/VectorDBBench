#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DIST_DIR=${DIST_DIR:-"$ROOT_DIR/dist"}
GOCACHE=${GOCACHE:-"$ROOT_DIR/.cache/go-build"}
TARGETS=${TARGETS:-"$(go env GOOS)/$(go env GOARCH) linux/amd64 linux/arm64 darwin/amd64 darwin/arm64"}

mkdir -p "$DIST_DIR" "$GOCACHE"

for target in $TARGETS; do
  goos=${target%/*}
  goarch=${target#*/}
  out="$DIST_DIR/s3vbench_${goos}_${goarch}"
  if [ "$goos" = "windows" ]; then
    out="${out}.exe"
  fi
  echo "building $target -> $out"
  (
    cd "$ROOT_DIR"
    GO111MODULE=on GOCACHE="$GOCACHE" CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" go build -trimpath -o "$out" ./cmd/s3vbench
  )
done
