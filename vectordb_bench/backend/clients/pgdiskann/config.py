from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, LiteralString, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgDiskANNConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class PgDiskANNConfig(DBConfig):
    user_name: SecretStr = SecretStr("postgres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> PgDiskANNConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class PgDiskANNIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True
    maintenance_work_mem: str | None
    max_parallel_workers: int | None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        if self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        return "vector_cosine_ops"

    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    def parse_metric_fun_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        if self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"

    @abstractmethod
    def index_param(self) -> dict: ...

    @abstractmethod
    def search_param(self) -> dict: ...

    @abstractmethod
    def session_param(self) -> dict: ...

    @staticmethod
    def _optionally_build_with_options(
        with_options: Mapping[str, Any],
    ) -> Sequence[dict[str, Any]]:
        """Walk through mappings, creating a List of {key1 = value} pairs. That will be used to build a where clause"""
        options = []
        for option_name, value in with_options.items():
            if value is not None:
                options.append(
                    {
                        "option_name": option_name,
                        "val": str(value),
                    },
                )
        return options

    @staticmethod
    def _optionally_build_set_options(
        set_mapping: Mapping[str, Any],
    ) -> Sequence[dict[str, Any]]:
        """Walk through options, creating 'SET 'key1 = "value1";' list"""
        session_options = []
        for setting_name, value in set_mapping.items():
            if value:
                session_options.append(
                    {
                        "parameter": {
                            "setting_name": setting_name,
                            "val": str(value),
                        },
                    },
                )
        return session_options


class PgDiskANNImplConfig(PgDiskANNIndexConfig):
    index: IndexType = IndexType.DISKANN
    max_neighbors: int | None
    l_value_ib: int | None
    l_value_is: float | None
    maintenance_work_mem: str | None = None
    max_parallel_workers: int | None = None

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "options": {
                "max_neighbors": self.max_neighbors,
                "l_value_ib": self.l_value_ib,
            },
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> dict:
        return {
            "diskann.l_value_is": self.l_value_is,
        }


_pgdiskann_case_config = {
    IndexType.DISKANN: PgDiskANNImplConfig,
}
