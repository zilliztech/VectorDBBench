# enVector with ANN (GAS) in VectorDBBench

This is a guide on how to use enVector with ANN index in VectorDBBench.

Basic usage of enVector with VectorDBBench follows the standard procedure for [VectorDBBench](https://github.com/zilliztech/VectorDBBench).

## Structure

```bash
.
├── centroids
│   └── embeddinggemma-300m
│       ├── centroids.npy             # centroids file for VCT
│       └── tree_info.pkl             # tree metadata for VCT
├── README_ENVECTOR.md
└── scripts
    ├── run_benchmark.sh              # benchmark script
    ├── envector_benchmark_config.yml # benchmark config file
    └── prepare_dataset.py            # download and prepare ground truth neighbors for dataset
```

## Prerequisites

### Install Python Dependencies
```bash
# 1. Create your environment
python -m venv .venv
source .venv/bin/activate

# 2. Install VectorDBBench
pip install -e .
pip install -r requirements.txt  # additional dependencies for VectorDBBench

# 3. Install es2
pip install es2==1.2.0a0
```

### Prepare dataset

Download dataset from huggingface and prepare ground truth neighbors.
For ANN benchmark, we provide two datasets:
- [PUBMED768D400K](https://huggingface.co/datasets/cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m)
- [BLOOMBERG768D378K](https://huggingface.co/datasets/cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m)

To prepare dataset, run the following command as example:

```bash
# Prepare dataset
python ./scripts/prepare_dataset.py \
    -d cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m
```

### Prepare enVector Server

To run enVector server with VCT index, please refer to the [enVector Deployment repository](https://github.com/CryptoLabInc/envector-deployment). As example, you can start the server with the following command:

```bash
# Start enVector server
./start_envector.sh --set VERSION_TAG=ann-v0.1
```

### Set Environment Variables

```bash
export DATASET_LOCAL_DIR="./dataset"
export NUM_PER_BATCH=500000  # set database size for efficiency
```

## Run Benchmark

See `./scripts/run_benchmark.sh` or `./scripts/envector_benchmark_config.yml` for an example of how to run benchmarks with enVector using VCT index, or use the following command:

```bash
python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --uri "localhost:50050" \
    --eval-mode mm \
    --case-type PerformanceCustomDataset \
    --db-label "PUBMED768D400K-IVF" \
    --custom-case-name PUBMED768D400K \
    --custom-dataset-name PUBMED768D400K \
    --custom-dataset-dir "" \
    --custom-dataset-size 400335 \
    --custom-dataset-dim 768 \
    --custom-dataset-file-count 1 \
    --custom-dataset-with-gt \
    --skip-custom-dataset-use-shuffled \
    --train-centroids True \
    --is-vct True \
    --centroids-path "./centroids/embeddinggemma-300m/centroids.npy" \
    --vct-path "./centroids/embeddinggemma-300m/tree_info.pkl" \
    --nlist 32768 \
    --nprobe 6
```