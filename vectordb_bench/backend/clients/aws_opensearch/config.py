import logging
from enum import Enum
from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

log = logging.getLogger(__name__)
class AWSOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 443
    user: str = ""
    password: SecretStr = ""

    def to_dict(self) -> dict:
        return {
            "hosts": [{'host': self.host, 'port': self.port}],
            "http_auth": (self.user, self.password.get_secret_value()),
            "use_ssl": True,
            "http_compress": True,
            "verify_certs": True,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 600,
        }


class AWSOS_Engine(Enum):
    nmslib = "nmslib"
    faiss = "faiss"
    lucene = "Lucene"


class AWSOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    engine: AWSOS_Engine = AWSOS_Engine.faiss
    efConstruction: int = 256
    efSearch: int = 256
    M: int = 16

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        elif self.metric_type == MetricType.COSINE:
            if self.engine == AWSOS_Engine.faiss:
                log.info(f"Using metric type as innerproduct because faiss doesn't support cosine as metric type for Opensearch")
                return "innerproduct"
            return "cosinesimil"
        return "l2"

    def index_param(self) -> dict:
        params = {
            "name": "hnsw",
            "space_type": self.parse_metric(),
            "engine": self.engine.value,
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
                "ef_search": self.efSearch
            }
        }
        return params

    def search_param(self) -> dict:
        return {}
