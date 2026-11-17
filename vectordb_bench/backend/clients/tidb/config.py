from enum import StrEnum
from typing import ClassVar, TypedDict

from pydantic import BaseModel, Field, SecretStr, model_validator

from ..api import DBCaseConfig, DBConfig, MetricType


class TiDBConfigDict(TypedDict):
    host: str
    port: int
    user: str
    password: str
    database: str
    ssl_verify_cert: bool
    ssl_verify_identity: bool


class TiDBConfig(DBConfig):
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset({"password"})

    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 4000
    db_name: str = "test"
    ssl: bool = False

    def to_dict(self) -> TiDBConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
            "database": self.db_name,
            "ssl_verify_cert": self.ssl,
            "ssl_verify_identity": self.ssl,
        }


class TiDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def get_metric_fn(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vec_l2_distance"
        if self.metric_type == MetricType.COSINE:
            return "vec_cosine_distance"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }

    def search_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }


class SPFreshBuildMode(StrEnum):
    INLINE = "inline"
    NON_INLINE = "non-inline"


class TiDBSPFreshIndexConfig(TiDBIndexConfig):
    spfresh_build_mode: SPFreshBuildMode = SPFreshBuildMode.INLINE
    spfresh_wait_timeout: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    min_partition_size: int | None = Field(default=None, ge=1)
    max_partition_size: int | None = Field(default=None, ge=2)
    write_beam_size: int | None = Field(default=None, ge=1, le=4096)
    search_beam_size: int | None = Field(default=None, ge=1, le=4096)
    oversample_factor: float | None = Field(default=None, ge=1, le=16, allow_inf_nan=False)
    hamming_factor: int | None = Field(default=None, ge=1, le=256)
    rabitq4_factor: int | None = Field(default=None, ge=1, le=256)

    @model_validator(mode="after")
    def validate_parameter_pairs(self):
        # Unspecified values come from the server (or the existing index).
        # Only compare pairs that the user explicitly supplied.
        if (
            self.min_partition_size is not None
            and self.max_partition_size is not None
            and self.min_partition_size > self.max_partition_size
        ):
            raise ValueError("min_partition_size must be <= max_partition_size")
        if (
            self.hamming_factor is not None
            and self.rabitq4_factor is not None
            and self.rabitq4_factor > self.hamming_factor
        ):
            raise ValueError("rabitq4_factor must be <= hamming_factor")
        return self

    def index_param(self) -> dict:
        return {
            **super().index_param(),
            **self.model_dump(
                include={"min_partition_size", "max_partition_size", "write_beam_size"}, exclude_none=True
            ),
        }

    def search_param(self) -> dict:
        return {
            **super().search_param(),
            **self.model_dump(
                include={"search_beam_size", "oversample_factor", "hamming_factor", "rabitq4_factor"}, exclude_none=True
            ),
        }
