from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


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
    metric_type: MetricType | None = None
    num_candidates: int | None = 1500  # Default numCandidates for vector search
    exact_search: bool = False  # Whether to use exact (ENN) search

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
                }
            ],
        }

    def search_param(self) -> dict:
        return {"numCandidates": self.num_candidates if not self.exact_search else None, "exact": self.exact_search}
