#!/bin/bash

set -euo pipefail

export DATASET_LOCAL_DIR="./dataset"

CENTROID_PATH=centroids/embeddinggemma-300m/centroids.npy
VCT_PATH=centroids/embeddinggemma-300m/tree_info.pkl
ENVECTOR_URI="localhost:50159"
REQUESTED_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            REQUESTED_TYPE="${2:-}"
            shift 2
            ;;
        --type=*)
            REQUESTED_TYPE="${1#--type=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

case "$REQUESTED_TYPE" in
    ""|flat|ivf) ;;
    *)
        echo "Invalid --type: $REQUESTED_TYPE (expected: flat or ivf)" >&2
        exit 1
        ;;
esac
COMMON_ARGS=(
    --uri "$ENVECTOR_URI"
    --eval-mode mm
    --case-type PerformanceCustomDataset
    --custom-case-name PUBMED768D400K
    --custom-dataset-name PUBMED768D400K
    --custom-dataset-dir ""
    --custom-dataset-size 400335
    --custom-dataset-dim 768
    --custom-dataset-file-count 1
    --custom-dataset-with-gt
    --skip-custom-dataset-use-shuffled
    --k 10
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

if [[ -z "$REQUESTED_TYPE" || "$REQUESTED_TYPE" == "flat" ]]; then
    run_case envectorflat "PUBMED768D400K-FLAT"
fi

if [[ -z "$REQUESTED_TYPE" || "$REQUESTED_TYPE" == "ivf" ]]; then
    export NUM_PER_BATCH=500000  # set database size for efficiency
    run_case envectorivfflat "PUBMED768D400K-IVF" \
        --is-vct True \
        --train-centroids True \
        --centroids-path "$CENTROID_PATH" \
        --vct-path "$VCT_PATH" \
        --nlist 32768 \
        --nprobe 6
fi
