from pydantic import BaseModel, SecretStr
from typing import ClassVar
from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

class HyperspaceDBConfig(DBConfig):
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset({"api_key"})
    
    host: str = "localhost:50051"
    api_key: SecretStr | None = None
    
    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "api_key": self.api_key.get_secret_value() if self.api_key else None
        }

class HyperspaceDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW
    metric_type: MetricType = MetricType.COSINE
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 100
    
    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        raise ValueError("Unsupported metric type: %s" % self.metric_type)
        
    def index_param(self) -> dict:
        return {
            "m": self.m,
            "ef_construction": self.ef_construction,
        }
        
    def search_param(self) -> dict:
        return {
            "ef_search": self.ef_search,
        }
