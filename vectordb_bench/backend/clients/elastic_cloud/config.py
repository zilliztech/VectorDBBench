from enum import Enum

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class ElasticCloudConfig(DBConfig, BaseModel):
    cloud_id: SecretStr
    password: SecretStr

    def to_dict(self) -> dict:
        return {
            "cloud_id": self.cloud_id.get_secret_value(),
            "basic_auth": ("elastic", self.password.get_secret_value()),
        }


class ESElementType(str, Enum):
    float = "float"  # 4 byte
    byte = "byte"  # 1 byte, -128 to 127


class ElasticCloudIndexConfig(BaseModel, DBCaseConfig):
    element_type: ESElementType = ESElementType.float
    index: IndexType = IndexType.ES_HNSW
    number_of_shards: int = 1
    number_of_replicas: int = 0
    refresh_interval: str = "30s"
    merge_max_thread_count: int = 8
    use_rescore: bool = False
    oversample_ratio: float = 2.0
    use_routing: bool = False
    use_force_merge: bool = True

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def __eq__(self, obj: any):
        return (
            self.index == obj.index
            and self.number_of_shards == obj.number_of_shards
            and self.number_of_replicas == obj.number_of_replicas
            and self.use_routing == obj.use_routing
            and self.efConstruction == obj.efConstruction
            and self.M == obj.M
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.index,
                self.number_of_shards,
                self.number_of_replicas,
                self.use_routing,
                self.efConstruction,
                self.M,
            )
        )

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        if self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        return {
            "type": "dense_vector",
            "index": True,
            "element_type": self.element_type.value,
            "similarity": self.parse_metric(),
            "index_options": {
                "type": self.index.value,
                "m": self.M,
                "ef_construction": self.efConstruction,
            },
        }

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }
