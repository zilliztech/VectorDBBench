import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class CSSConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 80
    user: str | None = None
    password: SecretStr | None = None
    http_compress: bool = False
    max_connection_pool_size: int = 100

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
            "http_compress": self.http_compress,
            "verify_certs": use_ssl,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 600,
            "maxsize":self.max_connection_pool_size, # Connection pool size for high QPS
            "max_retries":3, # Retry failed requests
        }

    @model_validator(mode="before")
    @classmethod
    def not_empty_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        skip = set(cls.common_short_configs()) | set(cls.common_long_configs()) | {"user", "password", "host"}
        for field_name, v in data.items():
            if field_name in skip:
                continue
            if isinstance(v, str) and not v:
                raise ValueError("Empty string!")
        return data

class CSS_Engine(Enum):
    hanns = "hanns"

class CSSIndexConfig(BaseModel, DBCaseConfig):
    model_config = ConfigDict(extra="ignore")
    metric_type: MetricType = MetricType.COSINE
    engine: CSS_Engine = CSS_Engine.hanns
    engine_name: str | None = None
    metric_type_name: str | None = None
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
    use_routing: bool = False # for label-filter cases
    replication_type: str | None = "DOCUMENT"
    knn_derived_source_enabled: bool = False
    memory_optimized_search: bool = False

    # HANNS build parameters
    max_degree: int | None = 56
    search_list_size_build: int | None = 200
    encoder: dict = Field(default_factory=lambda: {"name": "sq8"})
    pca_dim: int = 0

    # HANNS search parameters
    search_list_size: int = 100


    @model_validator(mode="before")
    @classmethod
    def validate_engine_name(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("engine_name"):
            engine_name = data["engine_name"].lower()
            if engine_name == "hanns":
                data["engine"] = CSS_Engine.hanns
            else:
                msg = f"Unknown engine_name: {engine_name}, only 'hanns' is supported"
                raise ValueError(msg)

        # Handle UI-passed encoder string and nbit
        encoder_str = data.get("encoder")
        if isinstance(encoder_str, str):
            if encoder_str == "sq8":
                data["encoder"] = {"name": "sq8"}
            elif encoder_str == "extended-rabitq":
                try:
                    nbit = int(data.get("nbit", 4))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"nbit must be an integer, got: {data.get('nbit')!r}") from e
                data["encoder"] = {"name": "extended-rabitq", "parameters": {"nbit": nbit}}
            else:
                log.warning(f"Unknown encoder: {encoder_str}, defaulting to sq8")
                data["encoder"] = {"name": "sq8"}
            # Remove nbit from data so pydantic doesn't complain about unknown field
            data.pop("nbit", None)

        return data

    def __eq__(self, obj: Any):
        if not isinstance(obj, CSSIndexConfig):
            return NotImplemented
        return (
            self.engine == obj.engine
            and self.max_degree == obj.max_degree
            and self.search_list_size_build == obj.search_list_size_build
            and self.encoder == obj.encoder
            and self.pca_dim == obj.pca_dim
            and self.search_list_size == obj.search_list_size
            and self.number_of_shards == obj.number_of_shards
            and self.number_of_replicas == obj.number_of_replicas
            and self.number_of_segments == obj.number_of_segments
            and self.use_routing == obj.use_routing
            and self.replication_type == obj.replication_type
            and self.knn_derived_source_enabled == obj.knn_derived_source_enabled
            and self.memory_optimized_search == obj.memory_optimized_search
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.engine,
                self.max_degree,
                self.search_list_size_build,
                tuple(self.encoder.items()) if self.encoder else (),
                self.pca_dim,
                self.search_list_size,
                self.number_of_shards,
                self.number_of_replicas,
                self.number_of_segments,
                self.use_routing,
                self.replication_type,
                self.knn_derived_source_enabled,
                self.memory_optimized_search,
            )
        )

    def parse_metric(self) -> str:
        if self.metric_type_name is not None and self.metric_type_name != "":
            log.info(f"User specified metric_type: {self.metric_type_name}")
            try:
                self.metric_type = MetricType[self.metric_type_name.upper()]
            except KeyError:
                raise ValueError(f"Invalid metric_type_name: '{self.metric_type_name}', valid values: {[m.name for m in MetricType]}")
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            return "cosinesimil"
        if self.metric_type == MetricType.L2:
            log.info("Using l2 as specified by user")
            return "l2"
        return "l2"

    def index_param(self) -> dict:
        space_type = self.parse_metric()
        log.info(
            f"CSS HANNS index config - "
            f"max_degree: {self.max_degree}, "
            f"search_list_size_build: {self.search_list_size_build}, "
            f"pca_dim: {self.pca_dim}, "
            f"encoder: {self.encoder}, "
            f"metric_type: {self.metric_type_name}, "
            f"space_type: {space_type}"
        )

        return {
            "name": "vamana",
            "engine": CSS_Engine.hanns.value,
            "space_type": space_type,
            "parameters": {
                "max_degree": self.max_degree,
                "search_list_size": self.search_list_size_build,
                "pca_dim": self.pca_dim,
                "encoder": self.encoder,
            },
        }

    def search_param(self) -> dict:
        return {
            "search_list_size": self.search_list_size,
        }
