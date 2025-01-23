from enum import Enum

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class QuantizationType(Enum):
    NONE = "none"
    BINARY = "binary"
    SCALAR = "scalar"


class MongoDBConfig(DBConfig, BaseModel):
    connection_string: SecretStr = "mongodb+srv://<user>:<password>@<cluster_name>.heatl.mongodb.net"
    database: str = "vdb_bench"

    def to_dict(self) -> dict:
        return {
            "connection_string": self.connection_string.get_secret_value(),
            "database": self.database,
        }


class MongoDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW  # MongoDB uses HNSW for vector search
    metric_type: MetricType = MetricType.COSINE
    num_candidates_ratio: int = 10  # Default numCandidates ratio for vector search
    quantization: QuantizationType = QuantizationType.NONE  # Quantization type if applicable

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "dotProduct"
        return "cosine"  # Default to cosine similarity

    def index_param(self) -> dict:
        return {
            "type": "vectorSearch",
            "fields": [
                {
                    "type": "vector",
                    "similarity": self.parse_metric(),
                    "numDimensions": None,  # Will be set in MongoDB class
                    "path": "vector",  # Vector field name
                    "quantization": self.quantization.value,
                }
            ],
        }

    def search_param(self) -> dict:
        return {"num_candidates_ratio": self.num_candidates_ratio}
