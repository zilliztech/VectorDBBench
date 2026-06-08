from typing import ClassVar

from pydantic import SecretStr, field_validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType
from ..milvus.config import DISKANNConfig, MilvusIndexConfig


class AliyunMilvusConfig(DBConfig):
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset({"user", "password"})

    uri: SecretStr = SecretStr("http://localhost:19530")
    user: str | None = None
    password: SecretStr | None = None
    num_shards: int = 1
    replica_number: int = 1
    # Tuning knobs for load performance
    load_reqs_size: int = int(1.5 * 1024 * 1024)
    # Controls when load_collection runs:
    # - False (default): load immediately after collection creation, then refresh_load in optimize().
    # - True: defer load until after compaction and index build in optimize().
    load_after_compaction: bool = False

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user if self.user else None,
            "password": self.password.get_secret_value() if self.password else None,
            "num_shards": self.num_shards,
            "replica_number": self.replica_number,
            "load_reqs_size": self.load_reqs_size,
            "load_after_compaction": self.load_after_compaction,
        }


class AliyunMilvusDISKANNConfig(DISKANNConfig):
    """AliyunMilvus DISKANN index config.

    Inherits from Milvus ``DISKANNConfig`` and adds Aliyun-specific fields.
    Overrides ``search_param`` to inject the three search-time knobs
    (``rerank_topk_multiplier`` / ``early_termination_threshold`` /
    ``cross_segment_rerank``) into the search ``params`` while keeping the
    original contract.
    """

    # ---- index build knobs ----
    max_degree: int = 48
    legacy: bool = False
    store_strategy: str = "MEMORY"
    quant_type: str = "RABITQ"
    num_threads: int = 4
    distance_strategy: str = "QUANT THEN MORE BITS"
    enable_prefetch: bool = False
    enable_thp: bool = False
    build_search_list: int = 200

    # ---- search-time knobs ----
    # Optional on purpose: a value of None means "do not send this search param"
    # (the server keeps its own default). They are only injected into search
    # params when explicitly set. Note ``0`` is a meaningful value (e.g.
    # rerank_topk_multiplier=0 disables rerank reads), so "unset" must be None,
    # not 0. UI passes a negative sentinel for "unset", normalized to None below.
    rerank_topk_multiplier: int | None = None
    early_termination_threshold: int | None = None
    cross_segment_rerank: bool | None = None

    @field_validator("rerank_topk_multiplier", "early_termination_threshold", mode="before")
    @classmethod
    def _normalize_optional_int(cls, v: object) -> int | None:
        if v is None or v == "" or v == "DEFAULT":
            return None
        iv = int(v)
        # negative value is the UI "unset" sentinel; 0 stays a real value
        return None if iv < 0 else iv

    @field_validator("cross_segment_rerank", mode="before")
    @classmethod
    def _normalize_optional_bool(cls, v: object) -> bool | None:
        if v is None or v == "" or v == "DEFAULT":
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        if self.metric_type == MetricType.COSINE:
            return MetricType.L2.value
        return self.metric_type.value

    def index_param(self) -> dict:
        extra_params: dict = {
            "max_degree": self.max_degree,
            "legacy": self.legacy,
            "store_strategy": self.store_strategy,
            "quant_type": self.quant_type,
            "num_threads": self.num_threads,
            "distance_strategy": self.distance_strategy,
            "search_list_size": self.build_search_list,
        }
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": extra_params,
        }

    def search_param(self) -> dict:
        # Only inject a search param when it was explicitly specified.
        params: dict = {}
        if self.search_list is not None:
            params["search_list"] = self.search_list
        if self.rerank_topk_multiplier is not None:
            params["rerank_topk_multiplier"] = self.rerank_topk_multiplier
        if self.early_termination_threshold is not None:
            params["early_termination_threshold"] = self.early_termination_threshold
        if self.cross_segment_rerank is not None:
            params["cross_segment_rerank"] = self.cross_segment_rerank
        return {
            "metric_type": self.parse_metric(),
            "params": params,
        }

    def load_param(self) -> dict:
        return {
            "knowhere.enable_thp": str(self.enable_thp).lower(),
            "knowhere.enable_prefetch": str(self.enable_prefetch).lower(),
        }


# Only DISKANN is supported by AliyunMilvus today. Other index types are not
# exposed here so callers fail fast (case_config_cls returns None) instead of
# silently using the upstream Milvus index implementation.
_aliyun_milvus_case_config: dict[IndexType, type[DBCaseConfig]] = {
    IndexType.DISKANN: AliyunMilvusDISKANNConfig,
}


__all__ = [
    "MilvusIndexConfig",
    "AliyunMilvusConfig",
    "AliyunMilvusDISKANNConfig",
    "_aliyun_milvus_case_config",
]
