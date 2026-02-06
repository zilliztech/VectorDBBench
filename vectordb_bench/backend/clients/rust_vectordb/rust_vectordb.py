"""
VectorDBBench client for Rust VectorDB

This client integrates the Rust VectorDB implementation with VectorDBBench
using PyO3 bindings for maximum performance.

Usage in VectorDBBench:
    Copy this directory to: VectorDBBench/vectordb_bench/backend/clients/rust_vectordb/
"""

import logging
from typing import Any, Tuple
from contextlib import contextmanager

# Import the Rust VectorDB Python bindings
try:
    from vectordb import PyVectorDB
except ImportError as e:
    raise ImportError(
        "vectordb Python package not found. "
        "Build it with: cd /path/to/vectordb && maturin develop --release --features python"
    ) from e

# Import VectorDBBench base classes
from ..api import VectorDB, DBCaseConfig, DBConfig, FilterOp

logger = logging.getLogger(__name__)


class RustVectorDB(VectorDB):
    """VectorDBBench client for Rust VectorDB
    
    This client provides a standardized interface for VectorDBBench to test
    the Rust VectorDB implementation with hierarchical clustering and RaBitQ quantization.
    """
    
    # Only support non-filtered searches for now
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    name: str = "RustVectorDB"
    
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: "RustVectorDBCaseConfig",
        collection_name: str = "benchmark",
        drop_old: bool = False,
        **kwargs
    ):
        """Initialize Rust VectorDB client
        
        Args:
            dim: Vector dimensionality
            db_config: Database configuration
            db_case_config: Case-specific configuration (index parameters)
            collection_name: Collection name (unused in this implementation)
            drop_old: Whether to drop old data (handled by creating new instance)
            **kwargs: Additional arguments
        """
        self.dim = dim
        self.collection_name = collection_name
        self.db_case_config = db_case_config
        self.db_config = db_config
        
        # Extract configuration
        self.branching_factor = db_case_config.branching_factor
        self.target_leaf_size = db_case_config.target_leaf_size
        self.probes = db_case_config.probes
        self.rerank_factor = db_case_config.rerank_factor
        self.metric = self._convert_metric(db_case_config.metric_type)
        
        # Temporary file path for mmap storage
        self.temp_file_path = db_config.get("temp_file_path", f"/tmp/vectordb_bench_{collection_name}.bin")
        
        logger.info(f"Initializing Rust VectorDB with dim={dim}, branching_factor={self.branching_factor}, "
                   f"target_leaf_size={self.target_leaf_size}, metric={self.metric}")
        
        # Create PyVectorDB instance
        try:
            self.db = PyVectorDB(
                dimension=dim,
                branching_factor=self.branching_factor,
                target_leaf_size=self.target_leaf_size,
                metric=self.metric,
                temp_file_path=self.temp_file_path,
            )
            logger.info("Rust VectorDB instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create Rust VectorDB: {e}")
            raise
        
        self._is_optimized = False
    
    def __getstate__(self):
        """Prepare object for pickling (multiprocessing support)
        
        PyO3 objects can't be pickled, so we save configuration and recreate
        the PyVectorDB instance after unpickling.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable PyVectorDB instance
        state['db'] = None
        return state
    
    def __setstate__(self, state):
        """Restore object after unpickling"""
        import os
        self.__dict__.update(state)
        # Recreate the PyVectorDB instance
        try:
            self.db = PyVectorDB(
                dimension=self.dim,
                branching_factor=self.branching_factor,
                target_leaf_size=self.target_leaf_size,
                metric=self.metric,
                temp_file_path=self.temp_file_path,
            )
            logger.debug(f"Recreated PyVectorDB instance after unpickling (dim={self.dim})")
            
            # If staging file exists, automatically rebuild the index in this subprocess
            # This supports both normal multiprocessing and skip-load scenarios
            staging_file = f"{self.temp_file_path}.staging"
            id_mapping_file = f"{self.temp_file_path}.idmap"
            
            # Load ID mapping if available
            if os.path.exists(id_mapping_file) and not hasattr(self, '_id_mapping'):
                try:
                    import pickle
                    with open(id_mapping_file, 'rb') as f:
                        self._id_mapping = pickle.load(f)
                    logger.info(f"Subprocess: loaded ID mapping ({len(self._id_mapping)} entries)")
                except Exception as e:
                    logger.warning(f"Subprocess: failed to load ID mapping: {e}")
            
            if os.path.exists(staging_file):
                logger.info(f"Subprocess: rebuilding index from staging file")
                try:
                    build_time = self.db.optimize()
                    self._is_optimized = True
                    logger.info(f"Subprocess: index rebuilt in {build_time:.2f}s")
                except Exception as e:
                    logger.error(f"Subprocess: failed to rebuild index: {e}")
                    raise  # Raise since we can't search without an index
                    
        except Exception as e:
            logger.error(f"Failed to recreate PyVectorDB after unpickling: {e}")
            raise
    
    def _convert_metric(self, metric_type: str) -> str:
        """Convert VectorDBBench metric type to Rust VectorDB metric"""
        metric_map = {
            "L2": "L2",
            "COSINE": "Cosine",
            "IP": "InnerProduct",
        }
        return metric_map.get(metric_type, "L2")
    
    @contextmanager
    def init(self):
        """Initialize database connection context
        
        For Rust VectorDB, this is a no-op since we're using in-process PyO3 bindings.
        The database instance is already created in __init__.
        """
        logger.debug("Entering Rust VectorDB context (no-op for in-process database)")
        yield
        logger.debug("Exiting Rust VectorDB context")
    
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int] | list[dict] | None = None,
        **kwargs: Any,
    ) -> Tuple[int, Exception | None]:
        """Insert embeddings into the database
        
        This accumulates vectors but doesn't build the index yet.
        The index is built during optimize().
        
        Args:
            embeddings: List of vectors to insert
            metadata: List of integer IDs for the vectors
            **kwargs: Additional arguments
            
        Returns:
            (count, error) - Number of vectors inserted and any error
        """
        try:
            # Store metadata IDs if provided
            if metadata is not None:
                if isinstance(metadata[0], dict):
                    # Some clients pass dicts, extract the ID
                    ids = [m.get('id', i) for i, m in enumerate(metadata)]
                else:
                    # VectorDBBench passes list of ints directly
                    ids = list(metadata)
                    
                # Store ID mapping for later lookup during search
                if not hasattr(self, '_id_mapping'):
                    self._id_mapping = []
                base_idx = len(self._id_mapping)
                self._id_mapping.extend(ids)
                logger.info(f"Stored ID mapping: batch {len(ids)} vectors, indices {base_idx}-{base_idx+len(ids)-1}, sample IDs: {ids[:3]} ... {ids[-3:] if len(ids) > 3 else []}")
                
                # Save ID mapping to file for subprocess access
                import pickle
                id_mapping_file = f"{self.temp_file_path}.idmap"
                with open(id_mapping_file, 'wb') as f:
                    pickle.dump(self._id_mapping, f)
                logger.debug(f"Saved ID mapping to {id_mapping_file}")
            
            count = self.db.insert_embeddings(embeddings)
            logger.debug(f"Inserted {count} embeddings (accumulated, not built yet)")
            return (count, None)
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            return (0, e)
    
    def optimize(self, data_size: int | None = None) -> None:
        """Build the hierarchical index from accumulated vectors
        
        This is called by VectorDBBench after all vectors are inserted
        and before search begins. It constructs the tree structure and
        quantizes vectors.
        
        Args:
            data_size: Optional hint about the data size (not used by Rust implementation)
        """
        try:
            logger.info("Building Rust VectorDB index...")
            build_time = self.db.optimize()
            self._is_optimized = True
            logger.info(f"Index built successfully in {build_time:.2f}s")
            
            # Log index statistics
            stats = self.db.get_stats()
            logger.info(f"Index stats: {stats['num_vectors']} vectors, "
                       f"depth={stats['max_depth']}, leaves={stats['num_leaves']}, "
                       f"memory={stats['memory_usage_bytes']/1024/1024:.1f}MB")
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
            raise
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Search for k nearest neighbors
        
        Args:
            query: Query vector
            k: Number of neighbors to return
            filters: Optional filters (not supported in this implementation)
            timeout: Optional timeout (not used)
            
        Returns:
            List of vector IDs sorted by distance (closest first)
        """
        if not self._is_optimized:
            raise RuntimeError("Index not optimized. Call optimize() first.")
        
        if filters is not None:
            logger.warning("Filters are not supported by Rust VectorDB, ignoring")
        
        try:
            # Rust search returns list of (array_index, distance) tuples
            results = self.db.search(
                query=query,
                k=k,
                probes=self.probes,
                rerank_factor=self.rerank_factor,
            )
            
            # Map array indices back to original IDs
            if hasattr(self, '_id_mapping') and self._id_mapping:
                # Use stored ID mapping
                logger.debug(f"Using ID mapping: {len(self._id_mapping)} entries, sample: array[0]->ID[{self._id_mapping[0]}]")
                return [self._id_mapping[array_idx] for array_idx, _distance in results]
            else:
                # Fallback: assume IDs == array indices (sequential from 0)
                logger.warning(f"No ID mapping found! Using array indices as IDs (fallback)")
                return [int(array_idx) for array_idx, _distance in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    @property
    def ready_to_load(self) -> bool:
        """Check if database is ready to accept inserts"""
        return True
    
    @property
    def ready_to_search(self) -> bool:
        """Check if database is ready for search queries"""
        return self._is_optimized
    
    def __repr__(self) -> str:
        return f"RustVectorDB(dim={self.dim}, vectors={len(self.db)})"


# Standalone test
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Rust VectorDB Python bindings...")
    
    # Create test data
    dim = 128
    num_vectors = 1000
    num_queries = 10
    k = 10
    
    print(f"Creating {num_vectors} random {dim}D vectors...")
    vectors = np.random.randn(num_vectors, dim).astype(np.float32).tolist()
    queries = np.random.randn(num_queries, dim).astype(np.float32).tolist()
    
    # Test configuration
    class TestConfig:
        branching_factor = 10
        target_leaf_size = 50
        probes = 5
        rerank_factor = 3
        metric_type = "L2"
    
    # Create client
    print("Creating RustVectorDB client...")
    client = RustVectorDB(
        dim=dim,
        db_config={},
        db_case_config=TestConfig(),
    )
    
    # Insert vectors
    print("Inserting vectors...")
    count, error = client.insert_embeddings(vectors)
    assert error is None, f"Insert failed: {error}"
    print(f"Inserted {count} vectors")
    
    # Build index
    print("Building index...")
    client.optimize()
    
    # Search
    print(f"Running {num_queries} searches...")
    for i, query in enumerate(queries):
        results = client.search_embedding(query, k=k)
        assert len(results) == k, f"Expected {k} results, got {len(results)}"
        print(f"Query {i}: Found {len(results)} neighbors, closest distance: {results[0][1]:.4f}")
    
    print("\n✅ All tests passed!")
