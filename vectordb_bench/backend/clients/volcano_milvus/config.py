from typing import ClassVar

from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType
from ..milvus.config import DISKANNConfig, MilvusIndexConfig


class VolcanoMilvusConfig(DBConfig):
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


class VolcanoMilvusDISKANNConfig(DISKANNConfig):
    """VolcanoMilvus DISKANN index config.

    Inherits from Milvus ``DISKANNConfig`` and adds Volcano-specific fields.
    Override ``index_param`` / ``search_param`` to inject Volcano-specific
    entries into ``params`` while keeping the original contract.
    """

    max_degree: int = 48
    legacy: bool = False
    store_strategy: str = "MEMORY"
    quant_type: str = "RABITQ"
    num_threads: int = 4
    distance_strategy: str = "QUANT THEN MORE BITS"
    enable_prefetch: bool = False
    enable_thp: bool = False
    build_search_list: int = 200

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

    def load_param(self) -> dict:
        return {
            "knowhere.enable_thp": str(self.enable_thp).lower(),
            "knowhere.enable_prefetch": str(self.enable_prefetch).lower(),
        }


# Only DISKANN is supported by VolcanoMilvus today. Other index types are not
# exposed here so callers fail fast (case_config_cls returns None) instead of
# silently using the upstream Milvus index implementation.
_volcano_milvus_case_config: dict[IndexType, type[DBCaseConfig]] = {
    IndexType.DISKANN: VolcanoMilvusDISKANNConfig,
}


__all__ = [
    "MilvusIndexConfig",
    "VolcanoMilvusConfig",
    "VolcanoMilvusDISKANNConfig",
    "_volcano_milvus_case_config",
]
