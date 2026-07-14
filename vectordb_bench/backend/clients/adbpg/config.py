from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class AdbpgSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class AdbpgConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API."""

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
                "options": "-c gp_session_role=utility",
            },
        }


class AdbpgIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    # ADB PG specific parameters
    hnsw_m: int = 48
    ef_search: int = 150
    ef_construction: int = 600
    nlist: int = 1024
    algorithm: str = "novamr"
    build_parallel_processes: int | None = None
    # rabitq quantization params
    rabitq_bits: int = 7
    quantize_rescore_amp: float = 0.0
    nova_adaptive_gamma: float = 0.0
    max_scan_points: int = 20000
    index_scan_mode: str = "snapshot"
    auto_reduction: bool = False
    pca_dim: int | None = None
    # novad-specific search param (no-op for novamr/HNSW algorithms)
    nprobe: int = 5

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.IP:
            return "ip"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)

    @staticmethod
    def _build_forced_set_options(set_mapping: Mapping[str, Any]) -> Sequence[dict[str, Any]]:
        """Always emit SET commands regardless of value (including 0 / 0.0)."""
        return [
            {
                "parameter": {
                    "setting_name": name,
                    "val": str(value),
                },
            }
            for name, value in set_mapping.items()
        ]

    def index_param(self) -> dict:
        with_options = [
            {"option_name": "algorithm", "val": self.algorithm},
            {"option_name": "hnsw_m", "val": self.hnsw_m},
            {"option_name": "hnsw_ef_construction", "val": self.ef_construction},
            {"option_name": "nlist", "val": self.nlist},
            {"option_name": "rabitq_bits", "val": self.rabitq_bits},
            # Covering index key length.
            {"option_name": "max_key_len", "val": 1},
        ]
        # Optional: auto_reduction=on — only include when True.
        # Uses raw=True so the value 'on' is emitted as a bare identifier
        # instead of a quoted string literal.
        if self.auto_reduction:
            with_options.append({"option_name": "auto_reduction", "val": "on", "raw": True})
        if self.pca_dim is not None:
            with_options.append({"option_name": "pca_dim", "val": self.pca_dim})

        return {
            "metric": self.parse_metric(),
            "build_parallel_processes": self.build_parallel_processes,
            "create_index_before_load": self.create_index_before_load,
            "create_index_after_load": self.create_index_after_load,
            "index_creation_with_options": with_options,
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
        }

    def session_param(self) -> AdbpgSessionCommands:
        # All CLI-driven search GUCs are always sent, regardless of value,
        # so that callers can explicitly tune any parameter — including to 0.
        session_parameters = {
            "fastann.quantize_rescore_amp": self.quantize_rescore_amp,
            "fastann.nova_adaptive_gamma": self.nova_adaptive_gamma,
            "fastann.hnsw_ef_search": self.ef_search,
            "fastann.hnsw_max_scan_points": self.max_scan_points,
            "fastann.index_scan_mode": self.index_scan_mode,
            "fastann.nova_nprobe": self.nprobe,
            "optimizer": "off",
            "elog_process_parameters": "off",
        }
        return {"session_options": self._build_forced_set_options(session_parameters)}
