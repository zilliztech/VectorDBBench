# VectorDBBench - AI Context Guide

> This file helps AI assistants understand and work with the VectorDBBench codebase effectively.

## Project Overview

**VectorDBBench** is an open-source benchmark tool for vector databases, maintained by Zilliz (creators of Milvus). It provides performance and cost-effectiveness comparisons across 40+ vector databases through GUI (Streamlit), CLI, and RESTful API interfaces.

- **Language**: Python 3.11+
- **License**: MIT
- **Repository**: https://github.com/zilliztech/VectorDBBench
- **Package**: `vectordb-bench` on PyPI

## Architecture

```
User Interface Layer
├── Streamlit Web UI (frontend/)     - Interactive benchmark runner
├── CLI Tool (cli/)                  - Command-line execution
└── RESTful API (restful/)           - HTTP API service

Core Business Logic (backend/)
├── BenchMarkRunner (interface.py)   - Task orchestration & async execution
├── CaseRunner (task_runner.py)      - Individual test case execution
├── Assembler (assembler.py)         - Task assembly from configs
├── Cases (cases.py)                 - Test case definitions
├── Dataset Manager (dataset.py)     - Dataset download & management
├── Runners (runner/)                - Execution engines (serial/concurrent/MP)
└── Results (models.py, metric.py)   - Metrics collection & persistence

Database Adapter Layer (backend/clients/)
├── api.py                           - VectorDB abstract base class
├── 40+ database implementations     - Milvus, PgVector, Elastic, Pinecone, etc.
└── __init__.py                      - DB enum & client registry

Data & Storage
├── S3 / AliyunOSS                   - Dataset source
├── Local parquet files              - Cached datasets
└── JSON results                     - Test output storage
```

## Directory Structure

```
VectorDBBench/
├── vectordb_bench/                  # Main package
│   ├── __init__.py                  # Global config (class config)
│   ├── __main__.py                  # GUI entry: launches Streamlit
│   ├── interface.py                 # Core API: BenchMarkRunner class
│   ├── models.py                    # Data models: TaskConfig, CaseResult, TestResult
│   ├── metric.py                    # Metrics: recall, ndcg, qps, latency
│   ├── base.py                      # BaseModel with common utilities
│   │
│   ├── backend/                     # Core benchmarking logic
│   │   ├── cases.py                 # CaseType enum, CapacityCase, PerformanceCase, StreamingPerformanceCase
│   │   ├── dataset.py               # Dataset enum, DatasetManager, DataSetIterator
│   │   ├── data_source.py           # S3/AliyunOSS dataset download
│   │   ├── filter.py                # Filter types: IntFilter, LabelFilter, NonFilter
│   │   ├── assembler.py             # Assembles TaskConfig into CaseRunners
│   │   ├── task_runner.py           # CaseRunner & TaskRunner execution logic
│   │   ├── result_collector.py      # Reads result JSON files
│   │   ├── utils.py                 # Utility functions
│   │   ├── clients/                 # Database client implementations
│   │   │   ├── api.py               # ABC: VectorDB, DBConfig, DBCaseConfig, MetricType, IndexType
│   │   │   ├── __init__.py          # DB enum, db2client mapping
│   │   │   ├── milvus/              # Milvus client
│   │   │   ├── zilliz_cloud/        # Zilliz Cloud client
│   │   │   ├── pgvector/            # PostgreSQL pgvector client
│   │   │   ├── elastic_cloud/       # Elastic Cloud client
│   │   │   ├── pinecone/            # Pinecone client
│   │   │   ├── qdrant_cloud/        # Qdrant client
│   │   │   ├── weaviate_cloud/      # Weaviate client
│   │   │   ├── redis/               # Redis client
│   │   │   ├── chroma/              # ChromaDB client
│   │   │   ├── mongodb/             # MongoDB client
│   │   │   ├── oceanbase/           # OceanBase client
│   │   │   └── ... (30+ more)
│   │   └── runner/                  # Execution engines
│   │       ├── serial_runner.py     # SerialInsertRunner, SerialSearchRunner
│   │       ├── concurrent_runner.py # ConcurrentInsertRunner
│   │       ├── mp_runner.py         # MultiProcessingSearchRunner
│   │       ├── read_write_runner.py # ReadWriteRunner (streaming)
│   │       ├── rate_runner.py       # Rate-controlled execution
│   │       └── executor.py          # Execution utilities
│   │
│   ├── cli/                         # Command-line interface
│   │   ├── vectordbbench.py         # CLI main entry
│   │   ├── cli.py                   # CommonTypedDict, run(), click decorators
│   │   └── batch_cli.py             # Batch execution from YAML config
│   │
│   ├── frontend/                    # Streamlit web UI
│   │   ├── vdbbench.py              # Streamlit entry point
│   │   ├── pages/                   # UI pages
│   │   │   ├── run_test.py          # Run benchmark page
│   │   │   ├── results.py           # Results display
│   │   │   ├── qps_recall.py        # QPS-Recall curves
│   │   │   ├── tables.py            # Table views
│   │   │   ├── custom.py            # Custom dataset page
│   │   │   └── ...
│   │   ├── components/              # Reusable UI components
│   │   └── config/                  # Frontend styling configs
│   │
│   ├── results/                     # Test result JSON storage
│   ├── config-files/                # Example YAML config files
│   └── custom/                      # Custom case configurations
│
├── tests/                           # Test suite
├── pyproject.toml                   # Project config, dependencies, entry points
├── install.py                       # Installation helper
├── Makefile                         # lint, format commands
└── Dockerfile                       # Container image
```

## Key Entry Points

| Entry | Command | File | Purpose |
|-------|---------|------|---------|
| GUI | `init_bench` | `__main__.py:run_streamlit()` | Launch Streamlit web UI |
| CLI | `vectordbbench [cmd] [opts]` | `cli/vectordbbench.py` | Command-line benchmark |
| REST | `init_bench_rest` | `restful/app.py` | Flask RESTful API |
| Batch | `vectordbbench batchcli --batch-config-file` | `cli/batch_cli.py` | Batch YAML execution |

## Key Classes & Abstractions

### VectorDB (backend/clients/api.py)
All database clients MUST implement this ABC:
- `__init__(dim, db_config, db_case_config, collection_name, drop_old)` - Initialize client
- `init()` (contextmanager) - Create/destroy connections safely
- `insert_embeddings(embeddings, metadata, labels_data)` - Insert vectors
- `search_embedding(query, k=100)` - Vector similarity search
- `optimize(data_size)` - Build index / optimize after insertion

### CaseType (backend/cases.py)
Enum defining all benchmark scenarios:
- `CapacityDim128/960` - Capacity/load tests
- `Performance768D100M/10M/1M` - Performance tests (various sizes)
- `Performance768D10M1P/99P` - Filtered search tests (1% / 99% filter rate)
- `StreamingPerformanceCase` - Streaming insert+search tests
- `LabelFilterPerformanceCase` - Label-based filtering tests
- `PerformanceCustomDataset` - User-defined dataset tests

### Dataset (backend/dataset.py)
Built-in datasets: SIFT, GIST, Cohere, OpenAI, LAION, Bioasq, Glove
- Downloaded from S3/AliyunOSS to `DATASET_LOCAL_DIR`
- Parquet format: `train.parquet`, `test.parquet`, `neighbors.parquet`

### BenchMarkRunner (interface.py)
Main orchestrator:
- `run(tasks, task_label)` - Submit benchmark tasks
- `get_results(result_dir)` - Retrieve historical results
- Uses `ProcessPoolExecutor` for async execution

## Data Flow

```
User Input (GUI/CLI)
    -> TaskConfig (models.py)
        -> Assembler.assemble_all() (assembler.py)
            -> TaskRunner with CaseRunner[] (task_runner.py)
                -> CaseRunner.run()
                    -> init_db()              # Create DB connection
                    -> dataset.prepare()      # Download dataset
                    -> _load_train_data()     # Concurrent insert
                    -> _optimize()            # Build index
                    -> _serial_search()       # Calculate recall/latency
                    -> _conc_search()         # Calculate QPS
                -> TestResult (models.py)
                    -> flush()                # Save JSON to results/
```

## Development Environment

This project uses `uv` for Python environment management. A `.venv` directory exists in the project root.

### Using the uv virtual environment

```bash
# Activate the venv
source .venv/bin/activate

# Or run commands directly via the venv Python
.venv/bin/python -m pytest tests/ -v

# Install dependencies with uv
uv pip install -e ".[test]"

# Install additional packages
uv pip install --python .venv/bin/python <package>
```

**Note:** The project requires Python 3.11+. The system `python` may be 3.10, so always use `.venv/bin/python` for running tests.

## Configuration

Environment variables (defined in `__init__.py`):
- `LOG_LEVEL` - Default: INFO
- `DATASET_SOURCE` - S3 or AliyunOSS
- `DATASET_LOCAL_DIR` - Default: /tmp/vectordb_bench/dataset
- `RESULTS_LOCAL_DIR` - Default: vectordb_bench/results
- `CONFIG_LOCAL_DIR` - Default: vectordb_bench/config-files
- `NUM_PER_BATCH` - Default: 100
- `NUM_CONCURRENCY` - Default: [1,5,10,20,30,40,60,80]
- `CONCURRENCY_DURATION` - Default: 30 (seconds)
- `DROP_OLD` - Default: True

## Adding a New Database Client

1. Create directory `backend/clients/mydb/`
2. Implement `config.py` with `MyDBConfig(DBConfig)` and `MyDBCaseConfig(DBCaseConfig)`
3. Implement `mydb.py` with `MyDB(VectorDB)` - 4 required methods
4. Register in `backend/clients/__init__.py` - add to DB enum and mappings
5. (Optional) Add CLI support with `cli.py` and register in `cli/vectordbbench.py`

## Common Tasks

### Run linting/formatting
```bash
make lint      # Check style
make format    # Auto-fix style
```

### Run a benchmark via CLI
```bash
# PgVector example
vectordbbench pgvectorhnsw \
  --host localhost --user-name postgres --password 'pass' \
  --db-name vectordb --case-type Performance768D10M \
  --m 16 --ef-construction 128 --ef-search 128

# Skip load, only search
vectordbbench pgvectorhnsw ... --skip-load --skip-drop-old

# Custom concurrency levels
vectordbbench pgvectorhnsw ... --num-concurrency 1,10,20,50
```

### Run GUI mode
```bash
init_bench
# or
python -m vectordb_bench
```

### Read results programmatically
```python
from vectordb_bench.interface import benchmark_runner
results = benchmark_runner.get_results()
```

## Important Notes

- **Python 3.11+ required**
- Uses `pydantic` v2 for all data models
- Uses `polars` (not pandas) for dataset reading
- Test cases support timeout controls (configurable per dataset size)
- Results are saved as JSON in `results/{db_name}/result_{date}_{label}_{db}.json`
- Supports custom datasets via Parquet format with specific column names
- Thread safety: set `thread_safe = False` on VectorDB subclass if client is not thread-safe

## Dependencies

Core: `click`, `streamlit`, `pydantic>=2.0`, `polars`, `plotly`, `pymilvus`, `scikit-learn`, `tqdm`

Optional (per database): `qdrant-client`, `pinecone`, `weaviate-client`, `elasticsearch`, `psycopg`, `redis`, `chromadb`, `pymongo`, etc.

## File Glossary

| File | Purpose |
|------|---------|
| `__init__.py` | Global configuration class, env var parsing |
| `interface.py` | BenchMarkRunner - main public API |
| `models.py` | TaskConfig, CaseConfig, TestResult, CaseResult |
| `metric.py` | Metric dataclass, recall/ndcg calculation |
| `backend/cases.py` | All test case definitions and CaseType enum |
| `backend/dataset.py` | Dataset definitions, DatasetManager, iterators |
| `backend/task_runner.py` | CaseRunner & TaskRunner execution |
| `backend/assembler.py` | TaskConfig -> CaseRunner assembly |
| `backend/clients/api.py` | VectorDB ABC, DBConfig, DBCaseConfig |
| `backend/clients/__init__.py` | DB enum, client registry mappings |
| `cli/cli.py` | CommonTypedDict, click parameter decorators |
| `frontend/vdbbench.py` | Streamlit app entry |

## CLI Quick Reference

### AWS S3 Vectors

S3 Vectors backend supports concurrent insert/search via boto3 with configurable tuning parameters.

```bash
# Minimum example — insert + search with default tuning
vectordbbench s3vectors \
  --access_key_id AKIA... \
  --secret_access_key ... \
  --bucket my-vector-bucket \
  --region us-west-2 \
  --index my-index \
  --metric cosine

# With tuning parameters (recommended for benchmark workloads)
vectordbbench s3vectors \
  --access_key_id AKIA... \
  --secret_access_key ... \
  --bucket my-vector-bucket \
  --insert-batch-size 200 \
  --max-pool-connections 50 \
  --retry-mode adaptive \
  --retry-max-attempts 10 \
  --metric cosine
```

**Tuning notes:**
- `--insert-batch-size`: AWS hard limit 500 vectors per PutVectors call. Default 100.
- `--max-pool-connections`: urllib3 pool size; should be >= 2x ConcurrentInsertRunner worker count. Default 50.
- `--retry-mode`: `adaptive` uses token-bucket backoff for ThrottlingException (recommended). Default `adaptive`.
- `--retry-max-attempts`: Total attempts including first call. Default 10.

### Milvus SPANN

```bash
# Minimum example
vectordbbench MilvusSPANN \
  --uri http://localhost:19530 \
  --collection my_spann_collection
```

### Milvus SPANN_RABITQ

```bash
# Minimum example
vectordbbench MilvusSPANNRaBitQ \
  --uri http://localhost:19530 \
  --collection my_spann_rabitq_collection
```

### Aliyun OSS Vector Bucket (AliOSS)

Aliyun OSS Vector Bucket SDK: `alibabacloud-oss-v2`. Backend at [vectordb_bench/backend/clients/alioss/](vectordb_bench/backend/clients/alioss/).

**CLI flags ([alioss/cli.py](vectordb_bench/backend/clients/alioss/cli.py)):**

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--access-key-id` | yes | — | Aliyun AccessKey ID |
| `--access-key-secret` | yes | — | Aliyun AccessKey Secret |
| `--account-id` | yes | — | 12-digit Aliyun account ID |
| `--bucket` | yes | — | Vector bucket name (must already exist) |
| `--region` | no | `cn-shenzhen` | Aliyun region |
| `--index` | no | `vdbbench-index` | Vector index name |
| `--metric` | no | `cosine` | `cosine` or `l2` |
| `--insert-batch-size` | no | `100` | PutVectors batch size; max 500 per Aliyun OSS docs |

**Bucket creation:** the backend does NOT auto-create the bucket — create it manually (Aliyun console or `client.put_vector_bucket()`) before first run. The backend DOES create/drop the index when `drop_old=True`.

#### No-filter: full benchmark (load + search)

```bash
vectordbbench alioss \
  --case-type Performance768D1M \
  --bucket my-vector-bucket \
  --access-key-id LTAI... \
  --access-key-secret ... \
  --account-id 1829... \
  --region cn-shenzhen \
  --metric cosine
```

#### No-filter: search-only against an existing 1M index

```bash
vectordbbench alioss \
  --case-type Performance768D1M \
  --skip-load \
  --bucket test \
  --index vdbbench1m \
  --num-concurrency 1 \
  --concurrency-duration 30 \
  --access-key-id LTAI... \
  --access-key-secret ... \
  --account-id 1829... \
  --region cn-shenzhen
```

`--skip-load` auto-disables `drop_old` ([cli.py:651](vectordb_bench/cli/cli.py#L651)) — `--skip-drop-old` is redundant.

#### Label filter benchmark (e.g. 1% / 10% selectivity)

Requires the `neighbors_labels_label_<N>p.parquet` ground-truth file alongside `shuffle_train.parquet`. Pre-download manually if vdbbench's S3 fetch times out:

```bash
# 1% ground truth
curl -L -o data_cache/cohere/cohere_medium_1m/neighbors_labels_label_1p.parquet \
  https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors_labels_label_1p.parquet

# 10% ground truth
curl -L -o data_cache/cohere/cohere_medium_1m/neighbors_labels_label_10p.parquet \
  https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors_labels_label_10p.parquet

# Symlink into vdbbench cache so S3 fetch is skipped
mkdir -p /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m
for f in shuffle_train.parquet test.parquet neighbors.parquet scalar_labels.parquet \
         neighbors_labels_label_1p.parquet neighbors_labels_label_10p.parquet; do
  ln -sf $(pwd)/data_cache/cohere/cohere_medium_1m/$f \
    /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/$f
done
```

**Full run (load + search) — first time only:**

```bash
vectordbbench alioss \
  --case-type LabelFilterPerformanceCase \
  --label-percentage 0.01 \
  --bucket vdbbench-labeled \
  --index vdbbench1m \
  --insert-batch-size 500 \
  --load-concurrency 4 \
  --num-concurrency 1 \
  --concurrency-duration 30 \
  --access-key-id LTAI... --access-key-secret ... --account-id 1829... \
  --region cn-shenzhen --metric cosine
```

**Search-only against an existing labeled index:**

```bash
# 1% label filter
vectordbbench alioss \
  --case-type LabelFilterPerformanceCase \
  --label-percentage 0.01 \
  --skip-load \
  --bucket vdbbench-labeled \
  --index vdbbench1m \
  --num-concurrency 1 \
  --concurrency-duration 30 \
  --access-key-id LTAI... --access-key-secret ... --account-id 1829... \
  --region cn-shenzhen --metric cosine

# 10% label filter (same index, different GT file)
vectordbbench alioss \
  --case-type LabelFilterPerformanceCase \
  --label-percentage 0.1 \
  --skip-load \
  --bucket vdbbench-labeled \
  --index vdbbench1m \
  --num-concurrency 1 \
  --concurrency-duration 30 \
  --access-key-id LTAI... --access-key-secret ... --account-id 1829... \
  --region cn-shenzhen --metric cosine
```

Mapping `--label-percentage` → corpus label:
- `0.01` → `label_1p` (1% of rows match)
- `0.1` → `label_10p` (10% of rows match)
- `0.001` → `label_0.1p`, `0.5` → `label_50p`, etc.

**Result location:** `vectordb_bench/results/AliOSS/result_<date>_<runid>_alioss.json`. Recall lives at `results[0].metrics.recall`. Quick read:

```bash
python -c "import json,sys; d=json.load(open(sys.argv[1])); m=d['results'][0]['metrics']; print({k:m[k] for k in ('recall','ndcg','qps','serial_latency_p99')})" \
  vectordb_bench/results/AliOSS/result_*.json
```

**Observed recall (Cohere 768D 1M, cn-shenzhen):**

| Case | recall | qps | serial_latency_p99 | Notes |
|------|--------|-----|-------------------|-------|
| No-filter | 0.9852 | 4.01 | 0.349s | ANN search on full 1M |
| Label-1% | 1.0 | 4.04 | 0.384s | ~10K hit set → brute-force exact |
| Label-10% | 1.0 | 2.94 | 0.374s | ~100K hit set → brute-force exact |

#### Standalone insert with checkpoint/resume

For multi-hour 1M+ inserts where vdbbench's load is too fragile (no resume on crash/network blip), use [ai_oss/insert_bench/insert_with_checkpoint.py](ai_oss/insert_bench/insert_with_checkpoint.py). Streams a parquet file into an Aliyun OSS Vector index in fixed-size batches and persists per-batch progress to a JSON file so you can `Ctrl-C` and re-run safely.

**Defaults:** bucket=`test`, index=`vdbbench1m`, total=1M, batch=100, dataset=`data_cache/cohere/cohere_medium_1m/shuffle_train.parquet`. Override via env vars.

```bash
# First run (creates checkpoint)
python ai_oss/insert_bench/insert_with_checkpoint.py

# Resume after Ctrl-C / crash / reboot — auto-skips inserted rows
python ai_oss/insert_bench/insert_with_checkpoint.py

# Clean restart (drops index + checkpoint)
DROP_OLD=1 python ai_oss/insert_bench/insert_with_checkpoint.py

# Faster: bigger batch (max 500 per Aliyun OSS docs)
BATCH_SIZE=500 python ai_oss/insert_bench/insert_with_checkpoint.py

# Smoke test on a small bucket
TOTAL_ROWS=1000 BUCKET=bench1ktest INDEX=bench1ktest \
  python ai_oss/insert_bench/insert_with_checkpoint.py
```

**Env vars:**

| Var | Default | Purpose |
|-----|---------|---------|
| `BUCKET` | `test` | Target vector bucket |
| `INDEX` | `vdbbench1m` | Target vector index |
| `TOTAL_ROWS` | `1000000` | How many rows to insert |
| `BATCH_SIZE` | `100` | PutVectors batch size (max 500) |
| `DROP_OLD` | `0` | `1` drops index and resets checkpoint |
| `PARQUET_PATH` | `data_cache/cohere/cohere_medium_1m/shuffle_train.parquet` | Source parquet |
| `LOG_EVERY_BATCHES` | `50` | Progress log cadence |
| `MAX_RETRIES` | `8` | Per-batch retry attempts on transient errors |
| `INITIAL_BACKOFF_S` | `2.0` | First retry delay |
| `MAX_BACKOFF_S` | `60.0` | Cap on exponential backoff |

**Checkpoint file:** `ai_oss/insert_bench/checkpoint_<bucket>_<index>.json`, atomic-replaced after each successful batch:

```json
{"next_index": 1000000, "inserted": 1000000, "bucket": "test", "index": "vdbbench1m", "ts": ...}
```

Resume reads `next_index`, fast-skips parquet batches whose end ≤ `next_index` without materializing them, partial-skips the straddling batch, then continues. Transient errors (DNS, timeout, 5xx, throttling) trigger inline exponential backoff (2s → 60s, 8 attempts). Hard failures save the checkpoint at the last known-good row and exit non-zero.

**Note:** the standalone script writes `metadata: {"id": str(rid)}` only — no scalar labels. For label filter cases, use the vdbbench load path instead (it auto-attaches labels from `scalar_labels.parquet`).

**Common pitfalls:**

- `'dict' object has no attribute 'key'` during search — `query_vectors` returns dicts, not objects. Already fixed at [alioss.py:167](vectordb_bench/backend/clients/alioss/alioss.py#L167).
- `Parquet magic bytes not found in footer` on load — `/tmp/vectordb_bench/dataset/.../shuffle_train.parquet` has correct byte size but corrupt content from a prior interrupted download. Replace with a symlink to `data_cache/`.
- `Read timeout on endpoint URL` for `assets.zilliz.com/...` — pre-download dataset files manually with `curl --retry`, then symlink into `/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/`.
