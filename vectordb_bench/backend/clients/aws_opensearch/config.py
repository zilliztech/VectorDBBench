import logging
from enum import Enum

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class AWSOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 443
    user: str = ""
    password: SecretStr = ""

    def to_dict(self) -> dict:
        return {
            "hosts": [{"host": self.host, "port": self.port}],
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
    efSearch: int = 100
    M: int = 16
    index_thread_qty: int | None = 4
    number_of_shards: int | None = 1
    number_of_replicas: int | None = 0
    number_of_segments: int | None = 1
    refresh_interval: str | None = "30s"
    force_merge_enabled: bool | None = True
    flush_threshold_size: str | None = "5120mb"
    number_of_indexing_clients: int | None = 1
    index_thread_qty_during_force_merge: int = 8
    cb_threshold: str | None = "50%"
    use_routing: bool = False  # for label-filter cases
    use_quant: bool = False
    oversample_factor: float = 1.0

    def __eq__(self, obj: any):
        return (
            self.engine == obj.engine
            and self.M == obj.M
            and self.efConstruction == obj.efConstruction
            and self.number_of_shards == obj.number_of_shards
            and self.number_of_replicas == obj.number_of_replicas
            and self.number_of_segments == obj.number_of_segments
            and self.use_routing == obj.use_routing
        )

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            if self.engine == AWSOS_Engine.faiss:
                log.info(
                    "Using innerproduct because faiss doesn't support cosine as metric type for Opensearch",
                )
                return "innerproduct"
            return "cosinesimil"
        return "l2"

    def index_param(self) -> dict:
        return {
            "name": "hnsw",
            "space_type": self.parse_metric(),
            "engine": self.engine.value,
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
                **(
                    {
                        "encoder": {
                            "name": "sq",
                        }
                    }
                    if self.use_quant
                    else {}
                ),
            },
        }

    def search_param(self) -> dict:
        return {"ef_search": self.efSearch}
