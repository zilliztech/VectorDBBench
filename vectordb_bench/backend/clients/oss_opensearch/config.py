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
    """In-memory scalar quantization types"""

    NONE = "None"
    LUCENE_SQ = "LuceneSQ"
    FAISS_SQFP16 = "FaissSQfp16"


# Compression level constants for disk-based mode
class CompressionLevel:
    """Valid compression levels for disk-based vector search"""

    LEVEL_1X = "1x"
    LEVEL_2X = "2x"
    LEVEL_4X = "4x"
    LEVEL_8X = "8x"
    LEVEL_16X = "16x"
    LEVEL_32X = "32x"

    ALL = [LEVEL_1X, LEVEL_2X, LEVEL_4X, LEVEL_8X, LEVEL_16X, LEVEL_32X]

    # Lucene: 1x, 4x | FAISS: 2x, 8x, 16x, 32x
    ENGINE_MAP = {
        LEVEL_1X: OSSOS_Engine.lucene,
        LEVEL_2X: OSSOS_Engine.faiss,
        LEVEL_4X: OSSOS_Engine.lucene,
        LEVEL_8X: OSSOS_Engine.faiss,
        LEVEL_16X: OSSOS_Engine.faiss,
        LEVEL_32X: OSSOS_Engine.faiss,
    }


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
    quantization_type: OSSOpenSearchQuantization = OSSOpenSearchQuantization.NONE
    confidence_interval: float | None = None
    clip: bool = False
    replication_type: str | None = "DOCUMENT"
    knn_derived_source_enabled: bool = False
    memory_optimized_search: bool = False
    on_disk: bool = False
    compression_level: str = CompressionLevel.LEVEL_32X
    oversample_factor: float = 1.0

    @validator("quantization_type", pre=True, always=True)
    def validate_quantization_type(cls, value: any):
        """Convert string values to enum"""
        if not value:
            return OSSOpenSearchQuantization.NONE

        if isinstance(value, OSSOpenSearchQuantization):
            return value

        mapping = {
            "None": OSSOpenSearchQuantization.NONE,
            "LuceneSQ": OSSOpenSearchQuantization.LUCENE_SQ,
            "FaissSQfp16": OSSOpenSearchQuantization.FAISS_SQFP16,
        }

        return mapping.get(value, OSSOpenSearchQuantization.NONE)

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
            and self.confidence_interval == obj.confidence_interval
            and self.clip == obj.clip
            and self.replication_type == obj.replication_type
            and self.knn_derived_source_enabled == obj.knn_derived_source_enabled
            and self.memory_optimized_search == obj.memory_optimized_search
            and self.on_disk == obj.on_disk
            and self.compression_level == obj.compression_level
            and self.oversample_factor == obj.oversample_factor
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
                self.confidence_interval,
                self.clip,
                self.replication_type,
                self.knn_derived_source_enabled,
                self.memory_optimized_search,
                self.on_disk,
                self.compression_level,
                self.oversample_factor,
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
        """Only use in-memory quantization when NOT in disk mode"""
        return not self.on_disk and self.quantization_type != OSSOpenSearchQuantization.NONE

    @property
    def resolved_engine(self) -> OSSOS_Engine:
        """Return engine based on mode: auto-selected for disk, configured for in-memory."""
        if self.on_disk:
            return CompressionLevel.ENGINE_MAP.get(self.compression_level, OSSOS_Engine.faiss)
        return self.engine

    def index_param(self) -> dict:
        resolved_engine = self.resolved_engine
        space_type = self.parse_metric()

        log.info(
            f"Index configuration - "
            f"mode: {'disk' if self.on_disk else 'in-memory'}, "
            f"configured_engine: {self.engine.value}, "
            f"resolved_engine: {resolved_engine.value}, "
            f"metric_type: {self.metric_type_name}, "
            f"space_type: {space_type}"
            f"{', ' if self.on_disk else ''}"
            f"{'compression_level: ' + self.compression_level if self.on_disk else ''}"
        )

        method_config = {
            "name": "hnsw",
            "engine": resolved_engine.value,
            "space_type": space_type,
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
            },
        }

        # Add encoder for in-memory quantization
        if self.use_quant:
            encoder_config = {"name": "sq"}

            if self.quantization_type == OSSOpenSearchQuantization.LUCENE_SQ:
                # Lucene SQ: optional confidence_interval
                if self.confidence_interval is not None:
                    encoder_config["parameters"] = {"confidence_interval": self.confidence_interval}

            elif self.quantization_type == OSSOpenSearchQuantization.FAISS_SQFP16 and self.clip:
                # FAISS SQfp16: optional clip parameter
                encoder_config["parameters"] = {"type": "fp16", "clip": True}

            method_config["parameters"]["encoder"] = encoder_config

        return method_config

    def search_param(self) -> dict:
        return {"ef_search": self.efSearch}
