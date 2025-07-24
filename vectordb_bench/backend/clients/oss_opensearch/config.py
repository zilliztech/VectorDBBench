import logging
from enum import Enum

from pydantic import BaseModel, SecretStr, root_validator, validator

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class OSSOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 80
    user: str | None = None
    password: SecretStr | None = None

    def to_dict(self) -> dict:
        use_ssl = self.port == 443
        http_auth = (
            (self.user, self.password.get_secret_value())
            if self.user is not None and self.password is not None and len(self.user) != 0 and len(self.password) != 0
            else ()
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

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if (
            field.name in cls.common_short_configs()
            or field.name in cls.common_long_configs()
            or field.name in ["user", "password", "host"]
        ):
            return v
        if isinstance(v, str | SecretStr) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class OSSOS_Engine(Enum):
    faiss = "faiss"
    lucene = "lucene"


class OSSOpenSearchQuantization(Enum):
    fp32 = "fp32"
    fp16 = "fp16"


class OSSOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    engine: OSSOS_Engine = OSSOS_Engine.faiss
    efConstruction: int = 256
    efSearch: int = 100
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
    index_thread_qty_during_force_merge: int = 8
    cb_threshold: str | None = "50%"
    number_of_indexing_clients: int | None = 1
    use_routing: bool = False  # for label-filter cases
    oversample_factor: float = 1.0
    quantization_type: OSSOpenSearchQuantization = OSSOpenSearchQuantization.fp32

    @root_validator
    def validate_engine_name(cls, values: dict):
        """Map engine_name string from UI to engine enum"""
        if values.get("engine_name"):
            engine_name = values["engine_name"].lower()
            if engine_name == "faiss":
                values["engine"] = OSSOS_Engine.faiss
            elif engine_name == "lucene":
                values["engine"] = OSSOS_Engine.lucene
            else:
                log.warning(f"Unknown engine_name: {engine_name}, defaulting to faiss")
                values["engine"] = OSSOS_Engine.faiss
        return values

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
        return self.quantization_type is not OSSOpenSearchQuantization.fp32

    def index_param(self) -> dict:
        log.info(f"Using engine: {self.engine} for index creation")
        log.info(f"Using metric_type: {self.metric_type_name} for index creation")
        log.info(f"Resulting space_type: {self.parse_metric()} for index creation")

        return {
            "name": "hnsw",
            "engine": self.engine.value,
            "space_type": self.parse_metric(),
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
                **(
                    {"encoder": {"name": "sq", "parameters": {"type": self.quantization_type.value}}}
                    if self.use_quant
                    else {}
                ),
            },
        }

    def search_param(self) -> dict:
        return {"ef_search": self.efSearch}
