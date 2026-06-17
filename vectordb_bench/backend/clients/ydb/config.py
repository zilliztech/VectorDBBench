import os
from typing import ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

from ..api import DBCaseConfig, DBConfig, MetricType


class YDBConfigDict(TypedDict, total=False):
    endpoint: str
    database: str
    auth_mode: str
    token: str
    user: str
    password: str
    table_name: str
    ssl_root_certificates_file: str
    auto_partitioning_min_partitions_count: int
    auto_partitioning_max_partitions_count: int
    auto_partitioning_table_partition_size_mb: int
    auto_partitioning_index_partition_size_mb: int
    operation_timeout_seconds: int


class YDBConfig(DBConfig):
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset(
        {"password", "token", "user", "table_name", "ssl_root_certificates_file"}
    )

    endpoint: str = "grpc://localhost:2136"
    database: str = "/local"
    auth_mode: str = "env"
    token: SecretStr | None = None
    user: str = ""
    password: SecretStr | None = None
    table_name: str = ""
    ssl_root_certificates_file: str = ""
    auto_partitioning_min_partitions_count: int = 1000
    auto_partitioning_max_partitions_count: int = 1100
    auto_partitioning_table_partition_size_mb: int = 1000
    auto_partitioning_index_partition_size_mb: int = 1000
    operation_timeout_seconds: int = 24 * 3600

    @model_validator(mode="after")
    def validate_partition_bounds(self) -> "YDBConfig":
        if self.auto_partitioning_min_partitions_count > self.auto_partitioning_max_partitions_count:
            msg = (
                "auto_partitioning_min_partitions_count must be less than or equal to "
                "auto_partitioning_max_partitions_count"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def apply_env_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if not data.get("endpoint") and os.environ.get("YDB_ENDPOINT"):
            data["endpoint"] = os.environ["YDB_ENDPOINT"]
        if not data.get("database") and os.environ.get("YDB_DATABASE"):
            data["database"] = os.environ["YDB_DATABASE"]
        if not data.get("ssl_root_certificates_file") and os.environ.get("YDB_SSL_ROOT_CERTIFICATES_FILE"):
            data["ssl_root_certificates_file"] = os.environ["YDB_SSL_ROOT_CERTIFICATES_FILE"]
        return data

    def to_dict(self) -> YDBConfigDict:
        token_str = self.token.get_secret_value() if self.token else ""
        password_str = self.password.get_secret_value() if self.password else ""
        result: YDBConfigDict = {
            "endpoint": self.endpoint,
            "database": self.database,
            "auth_mode": self.auth_mode,
            "token": token_str,
            "user": self.user,
            "password": password_str,
            "auto_partitioning_min_partitions_count": self.auto_partitioning_min_partitions_count,
            "auto_partitioning_max_partitions_count": self.auto_partitioning_max_partitions_count,
            "auto_partitioning_table_partition_size_mb": self.auto_partitioning_table_partition_size_mb,
            "auto_partitioning_index_partition_size_mb": self.auto_partitioning_index_partition_size_mb,
            "operation_timeout_seconds": self.operation_timeout_seconds,
        }
        if self.table_name:
            result["table_name"] = self.table_name
        if self.ssl_root_certificates_file:
            result["ssl_root_certificates_file"] = self.ssl_root_certificates_file
        return result


def index_on_columns(filters: Filter, *, with_scalar_labels: bool = False) -> tuple[str, ...]:
    if filters.type == FilterOp.NumGE:
        return ("id", "embedding")
    if with_scalar_labels or filters.type == FilterOp.StrEqual:
        return ("labels", "embedding")
    return ("embedding",)


class YDBIndexConfig(BaseModel, DBCaseConfig):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    metric_type: MetricType | None = None
    create_index_after_load: bool = True
    level: int | None = Field(default=None, alias="levels")
    nlist: int | None = Field(default=None, alias="clusters")
    num_leaves_to_search: int = Field(default=40, alias="kmeans_tree_search_top_size")
    overlap_clusters: int | None = 3
    cover_embedding: bool = True

    @field_validator("level", "nlist", mode="before")
    @classmethod
    def zero_means_auto(cls, value: int | None) -> int | None:
        if value == 0:
            return None
        return value

    @model_validator(mode="before")
    @classmethod
    def drop_empty_optional_numbers(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for key in ("level", "levels", "nlist", "clusters", "overlap_clusters"):
            if key in data and data.get(key) in ("", 0, None):
                data[key] = None
        return data

    def index_strategy(self) -> str:
        if self.metric_type == MetricType.L2:
            return "distance=euclidean"
        if self.metric_type == MetricType.IP:
            return "similarity=inner_product"
        return "similarity=cosine"

    def knn_function(self) -> str:
        if self.metric_type == MetricType.L2:
            return "EuclideanDistance"
        if self.metric_type == MetricType.IP:
            return "InnerProductSimilarity"
        return "CosineSimilarity"

    def sort_order(self) -> str:
        if self.metric_type in (MetricType.L2,):
            return "ASC"
        if self.metric_type == MetricType.IP:
            return "DESC"
        return "DESC"

    def index_on_columns(self, filters: Filter = non_filter, *, with_scalar_labels: bool = False) -> tuple[str, ...]:
        return index_on_columns(filters, with_scalar_labels=with_scalar_labels)

    def cover_clause(self, *, with_scalar_labels: bool = False) -> str:
        if with_scalar_labels or self.cover_embedding:
            return "COVER (embedding)"
        return ""

    def index_param(self, filters: Filter = non_filter, *, with_scalar_labels: bool = False) -> dict:
        on_columns = self.index_on_columns(filters, with_scalar_labels=with_scalar_labels)
        return {
            "strategy": self.index_strategy(),
            "levels": self.level,
            "clusters": self.nlist,
            "overlap_clusters": self.overlap_clusters,
            "on_columns": on_columns,
            "cover_clause": self.cover_clause(with_scalar_labels=with_scalar_labels),
        }

    def search_param(self) -> dict:
        return {
            "knn_function": self.knn_function(),
            "sort_order": self.sort_order(),
            "kmeans_tree_search_top_size": self.num_leaves_to_search,
        }
