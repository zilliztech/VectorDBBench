# enVector with ANN (GAS) in VectorDBBench

This guide demonstrates how to use enVector with an ANN index in VectorDBBench.

Basic usage of enVector with VectorDBBench follows the standard procedure for [VectorDBBench](https://github.com/zilliztech/VectorDBBench).

## Structure

```bash
.
├── centroids
│   └── embeddinggemma-300m
│       ├── centroids.npy             # centroids file for ANN
│       └── tree_info.pkl             # tree metadata for ANN
├── dataset
│   └── pubmed768d400k                # VectorDB ANN benchmark dataset
│       ├── neighbors.parquet
│       ├── test.npy
│       └── train.pkl
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

# 3. Install es2
pip install es2==1.2.0a1
```

### Prepare dataset

Prepare the following artifacts for the ANN benchmark with `scripts/prepare_dataset.py`:

- download datasets from HuggingFace
- prepare ground-truth neighbors
- download centroids and tree metadata for the GAS index for corresponding to the embedding model

For the ANN benchmark, we provide two datasets via HuggingFace:
- PUBMED768D400K: [cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m](https://huggingface.co/datasets/cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m)
- BLOOMBERG768D368K: [cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m](https://huggingface.co/datasets/cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m)

Also, we provide centroids and tree metadata for the corresponding embedding model used in the ANN benchmark:
- GASCENTROIDS: [cryptolab-playground/gas-centroids](https://huggingface.co/datasets/cryptolab-playground/gas-centroids)

To prepare dataset, run the following command as example:

```bash
# Prepare dataset
python ./scripts/prepare_dataset.py \
    -d cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m \
    -e embeddinggemma-300m
```

### Prepare enVector Server

To run enVector server with GAS index, please refer to the [enVector Deployment repository](https://github.com/CryptoLabInc/envector-deployment). For example, you can start the server with the following command:

```bash
# Start enVector server
git clone https://github.com/CryptoLabInc/envector-deployment
cd envector-deployment/docker-compose 
./start_envector.sh
```

### Set Environment Variables

```bash
# Set environment variables
export DATASET_LOCAL_DIR="./dataset"
export NUM_PER_BATCH=4096
```

## Run Benchmark

Refer to `./scripts/run_benchmark.sh` or `./scripts/envector_benchmark_config.yml` for benchmarks with enVector with ANN (VCT), or use the following command:

```bash
export NUM_PER_BATCH=500000 # set to the database size for efficiency with IVF_FLAT
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