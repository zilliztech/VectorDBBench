from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SpannerConfig:
    project_id: str
    instance_id: str
    database_id: str
    table_name: str = "vector_embeddings"
    emulator_host: Optional[str] = field(default=None)

from vectordb_bench.backend.clients.api import DBCaseConfig

@dataclass
class SpannerCaseConfig(DBCaseConfig):
    num_leaves: int = 1000
    num_leaves_to_search: int = 100
    tree_depth: int = 2 # Default value for tree_depth

    # Note: These parameters are intended for use with specific Spanner features
    # (e.g., vector indexes with OPTIONS, APPROX_COSINE_DISTANCE function)
    # which might require specific versions or configurations of Spanner.

    def index_param(self) -> dict:
        return {
            "num_leaves": self.num_leaves,
            "tree_depth": self.tree_depth,
        }

    def search_param(self) -> dict:
        return {
            "num_leaves_to_search": self.num_leaves_to_search,
        }
