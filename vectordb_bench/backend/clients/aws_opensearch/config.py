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
    s3vector = "s3vector"


class AWSOSQuantization(Enum):
    fp32 = "fp32"
    fp16 = "fp16"


class AWSOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    engine: AWSOS_Engine = AWSOS_Engine.faiss
    efConstruction: int | None = 256
    ef_search: int | None = 100
    engine_name: str | None = None
    metric_type_name: str | None = None
    M: int | None = 16
    index_thread_qty: int | None = 4
    number_of_shards: int | None = 1
    number_of_replicas: int | None = 0
    number_of_segments: int | None = 1
    refresh_interval: str | None = "60s"
    force_merge_enabled: bool | None = True
    flush_threshold_size: str | None = "5120mb"
    index_thread_qty_during_force_merge: int = 8
    cb_threshold: str | None = "50%"
    number_of_indexing_clients: int | None = 1
    use_routing: bool = False  # for label-filter cases
    oversample_factor: float = 1.0
    quantization_type: AWSOSQuantization = AWSOSQuantization.fp32

    def __eq__(self, obj: any):
        return (
            self.engine == obj.engine
            and self.M == obj.M
            and self.efConstruction == obj.efConstruction
            and self.number_of_shards == obj.number_of_shards
            and self.number_of_replicas == obj.number_of_replicas
            and self.number_of_segments == obj.number_of_segments
            and self.use_routing == obj.use_routing
            and self.quantization_type == obj.quantization_type
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.engine,
                self.M,
                self.efConstruction,
                self.number_of_shards,
                self.number_of_replicas,
                self.number_of_segments,
                self.use_routing,
                self.quantization_type,
            )
        )

    def parse_metric(self) -> str:
        log.info(f"User specified metric_type: {self.metric_type_name}")

        # Handle None or empty metric_type_name
        if self.metric_type_name is None or self.metric_type_name == "":
            log.info("No metric_type_name specified, defaulting to l2")
            self.metric_type = MetricType.L2
            return "l2"

        self.metric_type = MetricType[self.metric_type_name.upper()]
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            return "cosinesimil"
        if self.metric_type == MetricType.L2:
            log.info("Using l2 as specified by user")
            return "l2"
        return "l2"

    @property
    def use_quant(self) -> bool:
        return self.quantization_type is not AWSOSQuantization.fp32

    def index_param(self) -> dict:
        log.info(f"Using engine: {self.engine} for index creation")
        log.info(f"Using metric_type: {self.metric_type_name} for index creation")
        space_type = self.parse_metric()
        log.info(f"Resulting space_type: {space_type} for index creation")

        # Handle s3vector engine with simplified configuration
        # For s3vector, space_type should be set at the vector field level, not in method
        if self.engine == AWSOS_Engine.s3vector:
            return {"engine": "s3vector"}

        parameters = {"ef_construction": self.efConstruction, "m": self.M}

        if self.engine == AWSOS_Engine.faiss and self.quantization_type == AWSOSQuantization.fp16:
            parameters["encoder"] = {"name": "sq", "parameters": {"type": "fp16"}}

        # For other engines (faiss, lucene), space_type is set at method level
        return {
            "name": "hnsw",
            "engine": self.engine.value,
            "space_type": space_type,
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
                "ef_search": self.ef_search,
                **(
                    {"encoder": {"name": "sq", "parameters": {"type": self.quantization_type.fp16.value}}}
                    if self.use_quant
                    else {}
                ),
            },
        }

    def search_param(self) -> dict:
        # s3vector engine doesn't use ef_search parameter
        if self.engine == AWSOS_Engine.s3vector:
            return {}

        return {"ef_search": self.ef_search}
