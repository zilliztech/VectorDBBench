import logging
from enum import Enum

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class AWSOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 80
    user: str = ""
    password: SecretStr = ""

    def to_dict(self) -> dict:
        use_ssl = self.port == 443
        http_auth = (
            (self.user, self.password.get_secret_value()) if len(self.user) != 0 and len(self.password) != 0 else ()
        )
        return {
            "hosts": [{"host": self.host, "port": self.port}],
            "http_auth": http_auth,
            "use_ssl": use_ssl,
            "http_compress": True,
            "verify_certs": use_ssl,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 600,
        }


class AWSOS_Engine(Enum):
    faiss = "faiss"
    lucene = "lucene"


class AWSOSQuantization(Enum):
    fp32 = "fp32"
    fp16 = "fp16"


class AWSOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    engine: AWSOS_Engine = AWSOS_Engine.faiss
    efConstruction: int = 256
    ef_search: int = 200
    engine_name: str | None = None
    metric_type_name: str | None = None
    M: int = 16
    index_thread_qty: int | None = 4
    number_of_shards: int | None = 1
    number_of_replicas: int | None = 0
    number_of_segments: int | None = 1
    refresh_interval: str | None = "60s"
    force_merge_enabled: bool | None = True
    flush_threshold_size: str | None = "5120mb"
    index_thread_qty_during_force_merge: int
    cb_threshold: str | None = "50%"
    quantization_type: AWSOSQuantization = AWSOSQuantization.fp32

    def parse_metric(self) -> str:
        log.info(f"User specified metric_type: {self.metric_type_name}")
        self.metric_type = MetricType[self.metric_type_name.upper()]
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            return "cosinesimil"
        if self.metric_type == MetricType.L2:
            log.info("Using l2 as specified by user")
            return "l2"
        return "l2"

    def index_param(self) -> dict:
        log.info(f"Using engine: {self.engine} for index creation")
        log.info(f"Using metric_type: {self.metric_type_name} for index creation")
        log.info(f"Resulting space_type: {self.parse_metric()} for index creation")

        parameters = {"ef_construction": self.efConstruction, "m": self.M}

        if self.engine == AWSOS_Engine.faiss and self.faiss_use_fp16:
            parameters["encoder"] = {"name": "sq", "parameters": {"type": "fp16"}}

        return {
            "name": "hnsw",
            "engine": self.engine.value,
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
                "ef_search": self.efSearch,
                **(
                    {"encoder": {"name": "sq", "parameters": {"type": self.quantization_type.fp16.value}}}
                    if self.quantization_type is not AWSOSQuantization.fp32
                    else {}
                ),
            },
        }

    def search_param(self) -> dict:
        return {}
