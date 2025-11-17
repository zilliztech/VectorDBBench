#!/bin/bash

set -euo pipefail

DATASET_DIR=/data/vectordb_bench/dataset/PUBMED768D400K
COMMON_ARGS=(
    --uri "localhost:50050"
    --eval-mode mm
    --case-type PerformanceCustomDataset
    --custom-case-name PUBMED768D400K
    --custom-dataset-name PUBMED768D400K
    --custom-dataset-dir "$DATASET_DIR"
    --custom-dataset-size 400335
    --custom-dataset-dim 768
    --custom-dataset-file-count 1
    --skip-custom-dataset-with-gt
)

run_case() {
    local engine=$1
    local label=$2
    shift 2
    python -m vectordb_bench.cli.vectordbbench "$engine" \
        "${COMMON_ARGS[@]}" \
        --db-label "$label" \
        "$@"
}

run_case envectorflat "PUBMED768D400K-FLAT"

export NUM_PER_BATCH=500000  # set database size for efficiency
run_case envectorivfflat "PUBMED768D400K-IVF" --nprobe 6
