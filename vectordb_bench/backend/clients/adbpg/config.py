from ast import literal_eval
from collections.abc import Mapping
from typing import Any, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, field_validator, model_validator
from yaml import safe_load

from ..api import DBCaseConfig, DBConfig, MetricType

AdbpgParameterValue: TypeAlias = str | int | float | bool | None

DEFAULT_BUILD_RELOPTIONS: dict[str, AdbpgParameterValue] = {
    "algorithm": "novamr",
    "hnsw_m": 48,
    "hnsw_ef_construction": 600,
    "rabitq_bits": 7,
    "auto_reduction": False,
}
DEFAULT_BUILD_GUCS: dict[str, AdbpgParameterValue] = {
    "fastann.build_parallel_processes": 32,
}
DEFAULT_OPTIMIZE_UDF_TIMEOUT = 600

LEGACY_BUILD_RELOPTIONS = {
    "algorithm": "algorithm",
    "hnsw_m": "hnsw_m",
    "ef_construction": "hnsw_ef_construction",
    "nlist": "nlist",
    "rabitq_bits": "rabitq_bits",
    "auto_reduction": "auto_reduction",
    "pca_dim": "pca_dim",
}
LEGACY_BUILD_GUCS = {"build_parallel_processes": "fastann.build_parallel_processes"}
LEGACY_SEARCH_GUCS = {
    "ef_search": "fastann.hnsw_ef_search",
    "max_scan_points": "fastann.hnsw_max_scan_points",
    "quantize_rescore_amp": "fastann.quantize_rescore_amp",
    "nova_adaptive_gamma": "fastann.nova_adaptive_gamma",
    "index_scan_mode": "fastann.index_scan_mode",
    "nprobe": "fastann.nova_nprobe",
}


class AdbpgConfigDict(TypedDict):
    """Keys passed directly to psycopg.connect()."""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class AdbpgConfig(DBConfig):
    user_name: SecretStr = SecretStr("tester")
    password: SecretStr = SecretStr("")
    host: str = "localhost"
    port: int = 5432
    db_name: str = "postgres"

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value() if isinstance(self.user_name, SecretStr) else self.user_name
        pwd_str = self.password.get_secret_value()
        return {
            "table_name": "vector",
            "connect_config": {
                "host": self.host,
                "port": self.port,
                "dbname": self.db_name,
                "user": user_str,
                "password": pwd_str,
            },
        }


def parse_adbpg_parameter_group(value: Any, field_name: str) -> Mapping[str, Any]:
    """Parse a structured CLI/UI value while preserving YAML scalar types."""
    if isinstance(value, str):
        try:
            value = literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                value = safe_load(value)
            except Exception as exc:
                msg = f"{field_name} must be valid YAML or JSON"
                raise ValueError(msg) from exc
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a mapping"
        raise ValueError(msg)  # noqa: TRY004 - Pydantic validators must raise ValueError.
    return value


class AdbpgParameterGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reloption: dict[str, AdbpgParameterValue] = Field(default_factory=dict)
    guc: dict[str, AdbpgParameterValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse_mapping(cls, value: Any) -> Any:
        return parse_adbpg_parameter_group(value, cls.__name__)

    @field_validator("reloption", "guc", mode="before")
    @classmethod
    def validate_parameter_map(cls, value: Any, info) -> Any:  # noqa: ANN001
        if not isinstance(value, Mapping):
            msg = f"{info.field_name} must be a mapping"
            raise ValueError(msg)  # noqa: TRY004 - Pydantic validators must raise ValueError.
        for name, setting in value.items():
            if not isinstance(name, str) or not name:
                msg = f"{info.field_name} parameter names must be non-empty strings"
                raise ValueError(msg)
            if setting is not None and not isinstance(setting, (str, int, float, bool)):
                msg = f"{info.field_name}.{name} must be a YAML scalar"
                raise ValueError(msg)
        return dict(value)


class AdbpgBuildParameters(AdbpgParameterGroup):
    reloption: dict[str, AdbpgParameterValue] = Field(default_factory=DEFAULT_BUILD_RELOPTIONS.copy)
    guc: dict[str, AdbpgParameterValue] = Field(default_factory=DEFAULT_BUILD_GUCS.copy)

    @model_validator(mode="after")
    def merge_defaults(self):
        self.reloption = {**DEFAULT_BUILD_RELOPTIONS, **self.reloption}
        self.guc = {**DEFAULT_BUILD_GUCS, **self.guc}
        return self


class AdbpgAutotuneParameters(BaseModel):
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    model_config = ConfigDict(extra="allow")

    topk: list[int]
    target_recall: list[float]
    n_samples: int = 200
    n_trials: int = 200
    timeout: int = DEFAULT_OPTIMIZE_UDF_TIMEOUT

    @model_validator(mode="before")
    @classmethod
    def parse_mapping(cls, value: Any) -> Any:
        value = parse_adbpg_parameter_group(value, cls.__name__)
        if "enable" in value:
            msg = "omit autotune_parameters to disable autotune; enable is not supported"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_autotune(self):
        if not self.topk or any(value <= 0 for value in self.topk):
            msg = "topk must contain positive integers"
            raise ValueError(msg)
        if not self.target_recall or any(value <= 0 or value > 1 for value in self.target_recall):
            msg = "target_recall must contain values in the interval (0, 1]"
            raise ValueError(msg)
        for name in ("n_samples", "n_trials", "timeout"):
            value = getattr(self, name)
            minimum = 0 if name == "n_samples" else 1
            if value < minimum:
                msg = f"{name} must be at least {minimum}"
                raise ValueError(msg)
        return self


class AdbpgIndexConfig(BaseModel, DBCaseConfig):
    model_config = ConfigDict(extra="forbid")

    metric_type: MetricType | None = None
    build_parameters: AdbpgBuildParameters = Field(default_factory=AdbpgBuildParameters)
    search_parameters: AdbpgParameterGroup = Field(default_factory=AdbpgParameterGroup)
    autotune_parameters: AdbpgAutotuneParameters | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_parameters(cls, value: Any) -> Any:
        """Keep existing UI payloads working while accepting structured config."""
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        cls._move_legacy_parameters(data, "build_parameters", "reloption", LEGACY_BUILD_RELOPTIONS)
        cls._move_legacy_parameters(data, "build_parameters", "guc", LEGACY_BUILD_GUCS)
        cls._move_legacy_parameters(data, "search_parameters", "guc", LEGACY_SEARCH_GUCS)
        return data

    @staticmethod
    def _move_legacy_parameters(
        data: dict[str, Any],
        group_name: str,
        parameter_type: str,
        mapping: Mapping[str, str],
    ) -> None:
        legacy = {name: data[name] for name in mapping if name in data}
        if not legacy:
            return

        group = data.get(group_name)
        if group is None:
            group = {}
        elif isinstance(group, BaseModel):
            group = group.model_dump()
        elif isinstance(group, Mapping):
            group = dict(group)
        else:
            return

        parameters = group.get(parameter_type)
        if parameters is None:
            parameters = {}
        elif isinstance(parameters, Mapping):
            parameters = dict(parameters)
        else:
            return

        for name, target in mapping.items():
            if name in legacy:
                parameters.setdefault(target, legacy[name])
                data.pop(name)
        group[parameter_type] = parameters
        data[group_name] = group

    @field_validator("autotune_parameters", mode="before")
    @classmethod
    def empty_autotune_is_disabled(cls, value: Any) -> Any:
        return None if value == "" else value

    @property
    def algorithm(self) -> str:
        value = self.build_parameters.reloption["algorithm"]
        return str(value)

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.IP:
            return "ip"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            **self.build_parameters.model_dump(mode="json"),
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            **self.search_parameters.model_dump(mode="json"),
        }
