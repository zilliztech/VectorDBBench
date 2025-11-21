# enVector with ANN (GAS) in VectorDBBench

This is a guide on how to use enVector with ANN index in VectorDBBench.

Basic usage of enVector with VectorDBBench follows the standard procedure for [VectorDBBench](https://github.com/zilliztech/VectorDBBench).

## Prerequisites

```bash
# 1. Create your environment
python -m venv .venv
source .venv/bin/activate

# 2. Install vectordbbench
pip install -e .

# 3. Install es2
pip install es2==1.2.0a0
```

## Structure

```bash
.
├── centroids
│   └── embeddinggemma-300m
│       ├── centroids.npy
│       └── tree_info.pkl
├── README_ENVECTOR.md
└── scripts
    ├── run_benchmark.sh              # benchmark script
    ├── envector_benchmark_config.yml # benchmark config file
    └── prepare_neighbors.py          # prepare ground truth neighbors for dataset
```

## Run Benchmark

See `./scripts/benchmark.sh` or `./scripts/envector_customdataset_config.yml` for an example of how to run benchmarks with enVector using VCT index, or use the following command:

```bash
#!/bin/bash
export NUM_PER_BATCH=500000  # set database size for efficiency
python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --uri "localhost:50159" \
    --eval-mode mm \
    --case-type PerformanceCustomDataset \
    --db-label "PUBMED768D400K-IVF" \
    --custom-case-name PUBMED768D400K \
    --custom-dataset-name PUBMED768D400K \
    --custom-dataset-dir /data/vectordb_bench/dataset/PUBMED768D400K \
    --custom-dataset-size 400335 \
    --custom-dataset-dim 768 \
    --custom-dataset-file-count 1 \
    --custom-dataset-with-gt \
    --skip-custom-dataset-use-shuffled \
    --train-centroids True \
    --is-vct True \
    --centroids-path /data/centroids/embeddinggemma-300m/centroids.npy \
    --vct-path /data/centroids/embeddinggemma-300m/tree_info.pkl \
    --nlist 32768 \
    --nprobe 6
```