from enum import Enum
from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType, IndexType


class AliyunElasticsearchConfig(DBConfig, BaseModel):
    #: Protocol in use to connect to the node
    scheme: str = "http"
    host: str = ""
    port: int = 9200
    user: str = "elastic"
    password: SecretStr

    def to_dict(self) -> dict:
        return {
            "hosts": [{'scheme': self.scheme, 'host': self.host, 'port': self.port}],
            "basic_auth": (self.user, self.password.get_secret_value()),
        }


class ESElementType(str, Enum):
    float = "float"  # 4 byte
    byte = "byte"  # 1 byte, -128 to 127


class AliyunElasticsearchIndexConfig(BaseModel, DBCaseConfig):
    element_type: ESElementType = ESElementType.float
    index: IndexType = IndexType.ES_HNSW  # ES only support 'hnsw'

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        params = {
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
        return params

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }
