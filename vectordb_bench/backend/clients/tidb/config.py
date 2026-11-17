from enum import StrEnum
from typing import ClassVar, TypedDict

from pydantic import BaseModel, Field, SecretStr, field_validator

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
    spfresh_vector_index_param: str | None = None
    spfresh_wait_timeout: float = Field(default=600.0, gt=0, allow_inf_nan=False)

    @field_validator("spfresh_vector_index_param")
    @classmethod
    def normalize_vector_index_param(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    def index_param(self) -> dict:
        return {
            **super().index_param(),
            "spfresh_build_mode": self.spfresh_build_mode.value,
            "spfresh_vector_index_param": self.spfresh_vector_index_param,
        }
