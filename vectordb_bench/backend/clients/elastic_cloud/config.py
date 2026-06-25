from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, SecretStr, model_validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class ElasticCloudConfig(DBConfig, BaseModel):
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset(
        {"cloud_id", "scheme", "host", "user", "user_name", "password"}
    )

    cloud_id: SecretStr | None = None
    scheme: str | None = None
    host: SecretStr | None = None
    port: int = 9200
    user: str | None = None
    user_name: str | None = None
    password: SecretStr | None = None
    use_ssl: bool = False
    verify_certs: bool = True

    @model_validator(mode="after")
    def _check_connection_target(self) -> "ElasticCloudConfig":
        has_cloud_id = bool(self.cloud_id and self.cloud_id.get_secret_value())
        if not has_cloud_id and not self.host:
            msg = "Either cloud_id or host must be set"
            raise ValueError(msg)
        return self

    def _auth_user(self) -> str:
        return self.user_name or self.user or "elastic"

    def to_dict(self) -> dict:
        if self.cloud_id and self.cloud_id.get_secret_value():
            if not self.password:
                msg = "password is required when cloud_id is set"
                raise ValueError(msg)
            return {
                "cloud_id": self.cloud_id.get_secret_value(),
                "basic_auth": (self._auth_user(), self.password.get_secret_value()),
            }

        if not self.host:
            msg = "Either cloud_id or host must be set"
            raise ValueError(msg)

        host = self.host.get_secret_value()
        if host.startswith(("http://", "https://")):
            url = host
        else:
            scheme = self.scheme or ("https" if self.use_ssl else "http")
            url = f"{scheme}://{host}:{self.port}"

        config = {
            "hosts": [url],
            "verify_certs": self.verify_certs,
        }
        if self.password:
            config["basic_auth"] = (self._auth_user(), self.password.get_secret_value())
        elif self.user_name or (self.user and self.user != "elastic"):
            msg = "password is required when user_name is set"
            raise ValueError(msg)
        return config


class ESElementType(StrEnum):
    float = "float"  # 4 byte
    byte = "byte"  # 1 byte, -128 to 127


class ElasticCloudIndexConfig(BaseModel, DBCaseConfig):
    element_type: ESElementType = ESElementType.float
    index: IndexType = IndexType.ES_HNSW
    number_of_shards: int = 1
    number_of_replicas: int = 0
    refresh_interval: str = "30s"
    merge_max_thread_count: int = 8
    use_rescore: bool = False
    oversample_ratio: float = 2.0
    use_routing: bool = False
    use_force_merge: bool = True

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def __eq__(self, obj: any):
        return (
            self.index == obj.index
            and self.number_of_shards == obj.number_of_shards
            and self.number_of_replicas == obj.number_of_replicas
            and self.use_routing == obj.use_routing
            and self.efConstruction == obj.efConstruction
            and self.M == obj.M
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.index,
                self.number_of_shards,
                self.number_of_replicas,
                self.use_routing,
                self.efConstruction,
                self.M,
                2,
            )
        )

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        if self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        return {
            "type": "dense_vector",
            "index": True,
            "element_type": self.element_type.value,
            "similarity": self.parse_metric(),
            "index_options": {
                "type": self.index.value,
                "m": self.M,
                "ef_construction": self.efConstruction,
            },
        }

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }


class ElasticCloudFtsConfig(BaseModel, DBCaseConfig):
    number_of_shards: int = 1
    number_of_replicas: int = 0
    refresh_interval: str = "30s"
    use_force_merge: bool = True
    metric_type: MetricType = MetricType.BM25
    bm25_k1: float | None = None
    bm25_b: float | None = None

    def apply_fts_manifest(
        self,
        bm25_params: dict[str, float],
        analyzer_params: dict,
    ) -> tuple[DBCaseConfig, dict]:
        updates = {}
        applied_bm25_params = {}

        if "k1" in bm25_params:
            updates["bm25_k1"] = bm25_params["k1"]
            applied_bm25_params["k1"] = bm25_params["k1"]
        if "b" in bm25_params:
            updates["bm25_b"] = bm25_params["b"]
            applied_bm25_params["b"] = bm25_params["b"]

        return self.model_copy(update=updates), {
            "applied_bm25_params": applied_bm25_params,
            "unapplied_bm25_params": {k: v for k, v in bm25_params.items() if k not in applied_bm25_params},
            "applied_analyzer_params": {},
            "unapplied_analyzer_params": dict(analyzer_params),
        }

    def index_param(self) -> dict:
        text_mapping = {"type": "text"}
        if self.bm25_k1 is not None or self.bm25_b is not None:
            text_mapping["similarity"] = "vdbbench_bm25"
        return {
            "properties": {
                "doc_id": {"type": "keyword"},
                "text": text_mapping,
            },
        }

    def search_param(self) -> dict:
        return {}

    def similarity_settings(self) -> dict:
        if self.bm25_k1 is None and self.bm25_b is None:
            return {}

        bm25_settings = {"type": "BM25"}
        if self.bm25_k1 is not None:
            bm25_settings["k1"] = self.bm25_k1
        if self.bm25_b is not None:
            bm25_settings["b"] = self.bm25_b
        return {"similarity": {"vdbbench_bm25": bm25_settings}}
