"""
Configuration classes for Rust VectorDB VectorDBBench client
"""

from pydantic import BaseModel
from typing import Optional

# Import VectorDBBench base config classes
from ..api import DBConfig, DBCaseConfig, MetricType


class RustVectorDBConfig(DBConfig):
    """Connection configuration for Rust VectorDB
    
    Since Rust VectorDB runs in-process via Python bindings,
    there's no need for connection strings or authentication.
    """
    
    temp_file_path: Optional[str] = "/tmp/vectordb_bench.bin"
    """Path for temporary mmap storage file"""
    
    def to_dict(self) -> dict:
        """Return config as dictionary"""
        return {
            "temp_file_path": self.temp_file_path,
        }
    
    class Config:
        arbitrary_types_allowed = True


class RustVectorDBCaseConfig(RustVectorDBConfig, DBCaseConfig):
    """Combined connection and index configuration for Rust VectorDB
    
    Inherits connection parameters from RustVectorDBConfig and
    adds index/search parameters.
    """
    
    branching_factor: int = 100
    """Number of clusters at each level of the hierarchical tree.
    
    Higher values create shallower, wider trees:
    - 50-100: Good for 1M vectors
    - 100-200: Good for 10M+ vectors
    """
    
    target_leaf_size: int = 100
    """Target number of vectors per leaf node.
    
    Smaller values create deeper trees with more precise clustering:
    - 50-100: Balanced
    - 100-200: Faster build, slightly lower recall
    - 30-50: Slower build, slightly higher recall
    """
    
    probes: int = 100
    """Number of clusters to explore during tree traversal (beam width).
    
    Higher values improve recall but increase latency:
    - 20-50: Low latency
    - 100-200: Balanced
    - 500-1000: High recall
    """
    
    rerank_factor: int = 10
    """Rerank factor for two-phase search.
    
    Number of candidates to rerank with full precision = k * rerank_factor.
    Higher values improve recall but increase latency:
    - 3-5: Low latency
    - 10-15: Balanced
    - 25-50: High recall
    """
    
    metric_type: str = "L2"
    """Distance metric: L2, COSINE, or IP (Inner Product)"""
    
    def to_dict(self) -> dict:
        """Return config as dictionary"""
        return {
            "branching_factor": self.branching_factor,
            "target_leaf_size": self.target_leaf_size,
            "probes": self.probes,
            "rerank_factor": self.rerank_factor,
            "metric_type": self.metric_type,
        }
    
    def parse_metric(self, metric_type: str = None) -> str:
        """Convert VectorDBBench metric type to Rust format"""
        if metric_type is None:
            metric_type = self.metric_type
        metric_map = {
            "L2": "L2",
            "COSINE": "Cosine",
            "IP": "InnerProduct",
        }
        return metric_map.get(metric_type, "L2")
    
    def index_param(self) -> dict:
        """Return index build parameters"""
        return {
            "branching_factor": self.branching_factor,
            "target_leaf_size": self.target_leaf_size,
            "metric": self.parse_metric(),
        }
    
    def search_param(self) -> dict:
        """Return search parameters"""
        return {
            "probes": self.probes,
            "rerank_factor": self.rerank_factor,
        }
    
    class Config:
        arbitrary_types_allowed = True


# Predefined configurations for common scenarios

class LowLatencyConfig(RustVectorDBCaseConfig):
    """Optimized for low latency"""
    branching_factor: int = 100
    target_leaf_size: int = 100
    probes: int = 50
    rerank_factor: int = 25


class BalancedConfig(RustVectorDBCaseConfig):
    """Balanced latency and recall"""
    branching_factor: int = 100
    target_leaf_size: int = 100
    probes: int = 85
    rerank_factor: int = 75


class HighRecallConfig(RustVectorDBCaseConfig):
    """Optimized for high recall"""
    branching_factor: int = 100
    target_leaf_size: int = 100
    probes: int = 100
    rerank_factor: int = 150
