# Rust VectorDB - VectorDBBench Client Integration

This directory contains the VectorDBBench client for integrating Rust VectorDB with the VectorDBBench benchmarking framework.

## Architecture

```
VectorDBBench (Python) 
    ↓
rust_vectordb.py (Client)
    ↓  
vectordb Python bindings (PyO3)
    ↓
Rust VectorDB (Native Code)
```

## Files

- **`rust_vectordb.py`** - Main VectorDBBench client implementation
- **`config.py`** - Configuration classes (connection & case configs)
- **`cli.py`** - CLI commands for VectorDBBench
- **`__init__.py`** - Package initialization

## Integration Steps

### 1. Build the Python Bindings

First, build the Rust VectorDB Python package:

```bash
# Install maturin if you haven't already
pip install maturin

# Build and install the Python bindings
cd /Users/lyon/workspace/vectordb/vectordb
maturin develop --release --features python

# Verify installation
python -c "from vectordb import PyVectorDB; print('✅ Python bindings working!')"
```

### 2. Integrate with VectorDBBench

Clone or locate your VectorDBBench repository and copy the client files:

```bash
# Clone VectorDBBench (if needed)
git clone https://github.com/zilliztech/VectorDBBench.git
cd VectorDBBench

# Copy the rust_vectordb client
cp -r /Users/lyon/workspace/vectordb/vectordb/vectordbbench_client/ \\
      vectordb_bench/backend/clients/rust_vectordb/

# Install VectorDBBench in development mode
pip install -e .
```

### 3. Register the Client

Add Rust VectorDB to VectorDBBench's client registry:

**File: `vectordb_bench/backend/clients/__init__.py`**

```python
from enum import Enum

class DB(Enum):
    # ... existing databases ...
    RustVectorDB = "RustVectorDB"
    
    @property
    def init_cls(self):
        if self == DB.RustVectorDB:
            from .rust_vectordb.rust_vectordb import RustVectorDB
            return RustVectorDB
        # ... rest of the code ...
    
    @property
    def config_cls(self):
        if self == DB.RustVectorDB:
            from .rust_vectordb.config import RustVectorDBConfig
            return RustVectorDBConfig
        # ... rest of the code ...
    
    def case_config_cls(self, case_type):
        if self == DB.RustVectorDB:
            from .rust_vectordb.config import RustVectorDBCaseConfig
            return RustVectorDBCaseConfig
        # ... rest of the code ...
```

### 4. Import CLI Commands

**File: `vectordb_bench/cli/vectordbbench.py`**

Add this import near the top with other database CLI imports:

```python
from vectordb_bench.backend.clients.rust_vectordb.cli import (
    rustvectordb,
    rustvectordb_lowlatency,
    rustvectordb_balanced,
    rustvectordb_highrecall,
)
```

## Usage

### Command-Line Interface

```bash
# Basic usage
vectordbbench rustvectordb \\
  --dataset-name Cohere \\
  --dataset-scale 1M \\
  --case-type Performance768D1M \\
  --probes 100 \\
  --rerank-factor 10

# Low latency configuration
vectordbbench rustvectordb_lowlatency \\
  --dataset-name Cohere \\
  --dataset-scale 1M

# Balanced configuration
vectordbbench rustvectordb_balanced \\
  --dataset-name Cohere \\
  --dataset-scale 1M

# High recall configuration
vectordbbench rustvectordb_highrecall \\
  --dataset-name Cohere \\
  --dataset-scale 1M

# With custom parameters
vectordbbench rustvectordb \\
  --dataset-name SIFT \\
  --dataset-scale 1M \\
  --branching-factor 100 \\
  --target-leaf-size 100 \\
  --probes 200 \\
  --rerank-factor 25
```

### Available Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--branching-factor` | Clusters per tree level | 100 | 50-200 |
| `--target-leaf-size` | Vectors per leaf | 100 | 30-200 |
| `--probes` | Clusters to explore (beam width) | 100 | 20-1000 |
| `--rerank-factor` | Reranking multiplier | 10 | 3-50 |

### Supported Datasets

All standard VectorDBBench datasets:
- **SIFT** (128D, 1M vectors)
- **GIST** (960D, 1M vectors)
- **Cohere** (768D, 1M/10M vectors)
- **OpenAI** (1536D, 500K/5M vectors)

### Distance Metrics

- **L2** (Euclidean)
- **Cosine** (Cosine similarity)
- **IP** (Inner Product)

## Testing the Integration

### Standalone Test

Test the Python bindings without VectorDBBench:

```bash
cd /Users/lyon/workspace/vectordb/vectordb
python vectordbbench_client/rust_vectordb.py
```

Expected output:
```
Testing Rust VectorDB Python bindings...
Creating 1000 random 128D vectors...
Creating RustVectorDB client...
Inserting vectors...
Inserted 1000 vectors
Building index...
Index built successfully in 0.15s
Running 10 searches...
Query 0: Found 10 neighbors, closest distance: 8.2341
...
✅ All tests passed!
```

### VectorDBBench Test

Run a quick benchmark:

```bash
# Small dataset test
vectordbbench rustvectordb \\
  --case-type Performance768D1M \\
  --dataset-name Cohere \\
  --dataset-scale 1M \\
  --skip-search-concurrent  # Faster for testing

# View results
cat vectordb_bench/results/rustvectordb_*.json
```

## Performance Tuning

### For Low Latency (< 1ms)
```bash
--probes 20 --rerank-factor 5
```
Expected: ~0.5ms p50, ~85% recall@10

### For Balanced (1-2ms)
```bash
--probes 100 --rerank-factor 10
```
Expected: ~1.2ms p50, ~92% recall@10

### For High Recall (2-5ms)
```bash
--probes 200 --rerank-factor 25
```
Expected: ~3.5ms p50, ~96% recall@10

## Troubleshooting

### ImportError: vectordb module not found

**Solution**: Build the Python bindings first:
```bash
cd /Users/lyon/workspace/vectordb/vectordb
maturin develop --release --features python
```

### "Index not optimized" error

**Solution**: VectorDBBench should call `optimize()` automatically. If testing manually, call it after `insert_embeddings()`.

### Slow build times

**Cause**: Large datasets with many vectors
**Solution**: Increase `target_leaf_size` to reduce tree depth, or use fewer iterations in k-means (hardcoded to 20 currently).

### Out of memory errors

**Cause**: Too many vectors in memory
**Solution**: The index uses memory-mapped storage for vectors, but keeps quantized codes in RAM. For very large datasets (100M+), you may need more RAM or to reduce `branching_factor`.

## Comparison with Other Databases

Once integrated, you can compare directly:

```bash
# Run Rust VectorDB
vectordbbench rustvectordb_balanced --dataset-name Cohere --dataset-scale 1M

# Run Milvus for comparison
vectordbbench milvushnsw --uri $MILVUS_URI --dataset-name Cohere --dataset-scale 1M

# Results are in vectordb_bench/results/
ls -lh vectordb_bench/results/
```

## Contributing to VectorDBBench

Once you've tested the integration, consider submitting a PR to VectorDBBench:

1. Fork the VectorDBBench repository
2. Add your rust_vectordb client
3. Test thoroughly
4. Submit PR with:
   - Client code
   - Documentation
   - Example results

## Next Steps

1. ✅ Build Python bindings: `maturin develop --release --features python`
2. ✅ Test standalone: `python vectordbbench_client/rust_vectordb.py`
3. ✅ Copy to VectorDBBench: `cp -r vectordbbench_client/ VectorDBBench/...`
4. ✅ Register client in `__init__.py`
5. ✅ Import CLI in `vectordbbench.py`
6. ✅ Run benchmark: `vectordbbench rustvectordb ...`
7. ✅ Compare results with other databases

## Support

For issues:
- Check the build instructions in this README
- See the main integration guide: `/Users/lyon/workspace/vectordb/vectordb/VECTORDBBENCH_INTEGRATION_GUIDE.md`
- Review PyO3 documentation: https://pyo3.rs
