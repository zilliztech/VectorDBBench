from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, LiteralString, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class AlloyDBConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class AlloyDBConfig(DBConfig):
    user_name: SecretStr = SecretStr("postgres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> AlloyDBConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class AlloyDBIndexParam(TypedDict):
    metric: str
    index_type: str
    index_creation_with_options: Sequence[dict[str, Any]]
    maintenance_work_mem: str | None
    max_parallel_workers: int | None


class AlloyDBSearchParam(TypedDict):
    metric_fun_op: LiteralString


class AlloyDBSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class AlloyDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.DP:
            return "dot_product"
        return "cosine"

    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    @abstractmethod
    def index_param(self) -> AlloyDBIndexParam: ...

    @abstractmethod
    def search_param(self) -> AlloyDBSearchParam: ...

    @abstractmethod
    def session_param(self) -> AlloyDBSessionCommands: ...

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


class AlloyDBScaNNConfig(AlloyDBIndexConfig):
    index: IndexType = IndexType.SCANN
    num_leaves: int | None
    quantizer: str | None
    enable_pca: str | None
    max_num_levels: int | None
    num_leaves_to_search: int | None
    max_top_neighbors_buffer_size: int | None
    pre_reordering_num_neighbors: int | None
    num_search_threads: int | None
    max_num_prefetch_datasets: int | None
    maintenance_work_mem: str | None = None
    max_parallel_workers: int | None = None

    def index_param(self) -> AlloyDBIndexParam:
        index_parameters = {
            "num_leaves": self.num_leaves,
            "max_num_levels": self.max_num_levels,
            "quantizer": self.quantizer,
        }
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(index_parameters),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "enable_pca": self.enable_pca,
        }

    def search_param(self) -> AlloyDBSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> AlloyDBSessionCommands:
        session_parameters = {
            "scann.num_leaves_to_search": self.num_leaves_to_search,
            "scann.max_top_neighbors_buffer_size": self.max_top_neighbors_buffer_size,
            "scann.pre_reordering_num_neighbors": self.pre_reordering_num_neighbors,
            "scann.num_search_threads": self.num_search_threads,
            "scann.max_num_prefetch_datasets": self.max_num_prefetch_datasets,
        }
        return {"session_options": self._optionally_build_set_options(session_parameters)}


_alloydb_case_config = {
    IndexType.SCANN: AlloyDBScaNNConfig,
}
