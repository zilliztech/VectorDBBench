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
    efSearch: int = 256
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
    number_of_indexing_clients: int | None = 1
    index_thread_qty_during_force_merge: int
    cb_threshold: str | None = "50%"
    faiss_use_fp16: bool | None = True
    
    def __init__(self, **data):
        super().__init__(**data)

        # self.faiss_use_fp16 = data.get("faiss_use_fp16", False) if self.engine == AWSOS_Engine.faiss else False
        if self.engine_name is not None:
            try:
                self.engine = AWSOS_Engine[self.engine_name.lower()]
                log.info(f"Setting engine from engine_name: {self.engine_name} -> {self.engine}")
            except (KeyError, ValueError):
                log.warning(f"Invalid engine name: {self.engine_name}, using default: {self.engine}")
                
        if self.metric_type_name is not None:
            try:
                self.metric_type = MetricType[self.metric_type_name.upper()]
                log.info(f"Setting metric_type from metric_type_name: {self.metric_type_name} -> {self.metric_type}")
            except (KeyError, ValueError):
                log.warning(f"Invalid metric type: {self.metric_type_name}, using default: {self.metric_type}")

    def parse_metric(self) -> str:
        log.info(f"User specified metric_type: {self.metric_type_name}")
        self.metric_type = MetricType[self.metric_type_name.upper()]
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            if self.engine == AWSOS_Engine.faiss:
                log.info("Using innerproduct because faiss doesn't support cosine as metric type for Opensearch")
                return "innerproduct"
            return "cosinesimil"
        if self.metric_type == MetricType.L2:
            log.info("Using l2 as specified by user")
            return "l2"
        return "l2"

    def index_param(self) -> dict:
        log.info(f"Using engine: {self.engine} for index creation")
        log.info(f"Using metric_type: {self.metric_type_name} for index creation")
        log.info(f"Resulting space_type: {self.parse_metric()} for index creation")

        parameters = {
            "ef_construction": self.efConstruction,
            "m": self.M
        }

        if self.engine == AWSOS_Engine.faiss and self.faiss_use_fp16:
            parameters["encoder"] = {
                "name": "sq",
                "parameters": {
                    "type": "fp16"
                }
            }

        return {
            "name": "hnsw",
            "engine": self.engine.value,
            "space_type": self.parse_metric(),
            "parameters": parameters
        }

    def search_param(self) -> dict:
        return {}