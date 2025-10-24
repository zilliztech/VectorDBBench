from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, LiteralString, TypedDict
from pydantic import BaseModel, SecretStr
from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class VexDBConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class VexDBConfig(DBConfig):
    user_name: SecretStr = "test_user1"
    password: SecretStr
    host: str = "192.168.232.128"
    port: int = 5432
    db_name: str = "sc"
    table_name: str = "vdbbench_table_test"

    def to_dict(self) -> VexDBConfigDict:
        user_str = self.user_name.get_secret_value() if isinstance(self.user_name, SecretStr) else self.user_name
        pwd_str = self.password.get_secret_value()
        return {
            "connect_config": {
                "host": self.host,
                "port": self.port,
                "dbname": self.db_name,
                "user": user_str,
                "password": pwd_str,
            },
            "table_name": self.table_name,
        }


class VexDBIndexParam(TypedDict):
    metric: str
    index_type: str
    index_creation_with_options: Sequence[dict[str, Any]]
    maintenance_work_mem: str | None
    max_parallel_workers: int | None


class VexDBSearchParam(TypedDict):
    metric_fun_op: LiteralString


class VexDBSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class VexDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    # create_index_before_load: bool = False
    create_index_after_load: bool = True

    def parse_metric(self) -> str:
        d = {
                MetricType.L2: "floatvector_l2_ops",
                MetricType.IP: "floatvector_ip_ops",
                MetricType.COSINE: "floatvector_cosine_ops",
        }
        return d.get(self.metric_type)

    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"


    @abstractmethod
    def index_param(self) -> VexDBIndexParam: ...

    @abstractmethod
    def search_param(self) -> VexDBSearchParam: ...

    @abstractmethod
    def session_param(self) -> VexDBSessionCommands: ...

    @staticmethod
    def _optionally_build_with_options(with_options: Mapping[str, Any]) -> Sequence[dict[str, Any]]:
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
    def _optionally_build_set_options(set_mapping: Mapping[str, Any]) -> Sequence[dict[str, Any]]:
        """Walk through options, creating 'SET 'key1 = "value1";' list"""
        session_options = []
        for setting_name, value in set_mapping.items():
            if value is not None:
                session_options.append(
                    {
                        "parameter": {
                            "setting_name": setting_name,
                            "val": str(value),
                        },
                    },
                )
        return session_options


class VexDBIVFFlatConfig(VexDBIndexConfig):
    """
    An IVFFlat index divides vectors into lists, and then searches a subset of those lists that are
    closest to the query vector. It has faster build times and uses less memory than HNSW,
    but has lower query performance (in terms of speed-recall tradeoff).

    Three keys to achieving good recall are:

    Create the index after the table has some data
    Choose an appropriate number of lists - a good place to start is rows / 1000 for up to 1M rows and sqrt(rows) for
    over 1M rows.
    When querying, specify an appropriate number of probes (higher is better for recall, lower is better for speed) -
    a good place to start is sqrt(lists)
    """

    lists: int | None
    probes: int | None
    index: IndexType = IndexType.ES_IVFFlat
    maintenance_work_mem: str | None = None
    max_parallel_workers: int | None = None
    create_index_before_load: bool = False

    def index_param(self) -> VexDBIndexParam:
        index_parameters = {"ivf_nlist": self.lists, "parallel_workers": self.max_parallel_workers}
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(index_parameters),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "create_index_before_load": self.create_index_before_load,
        }

    def search_param(self) -> VexDBSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> VexDBSessionCommands:
        session_parameters = {"ivf_probes": self.probes}
        return {"session_options": self._optionally_build_set_options(session_parameters)}


class VexDBHNSWConfig(VexDBIndexConfig):
    """
    An HNSW index creates a multilayer graph. It has better query performance than IVFFlat (in terms of
    speed-recall tradeoff), but has slower build times and uses more memory. Also, an index can be
    created without any data in the table since there isn't a training step like IVFFlat.
    """

    m: int | None  # DETAIL:  Valid values are between "2" and "100".
    ef_construction: int | None  # ef_construction must be greater than or equal to 2 * m
    ef_search: int | None
    index: IndexType = IndexType.ES_HNSW
    maintenance_work_mem: str | None = None
    max_parallel_workers: int | None = None
    create_index_before_load: bool = False


    def index_param(self) -> VexDBIndexParam:
        index_parameters = {"m": self.m, "ef_construction": self.ef_construction, "parallel_workers": self.max_parallel_workers}
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(index_parameters),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "create_index_before_load": self.create_index_before_load,
        }

    def search_param(self) -> VexDBSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> VexDBSessionCommands:
        session_parameters = {"hnsw_ef_search": self.ef_search}
        return {"session_options": self._optionally_build_set_options(session_parameters)}

class VexDBHybridANNConfig(VexDBIndexConfig):
    m: int | None  # DETAIL:  Valid values are between "2" and "100".
    ef_construction: int | None  # ef_construction must be greater than or equal to 2 * m
    ef_search: int | None
    index: IndexType = IndexType.HybridAnn
    maintenance_work_mem: str | None = None
    max_parallel_workers: int | None = None
    create_index_before_load: bool = False
    graph_magnitude_threshold: int | None = None
    vec_index_magnitudes: str | None = None
    hybrid_query_ivf_probes_factor: int | None = None
    col_name_list: str | None


    def index_param(self) -> VexDBIndexParam:
        index_parameters = {"m": self.m, "ef_construction": self.ef_construction, "parallel_workers": self.max_parallel_workers, "graph_magnitude_threshold": self.graph_magnitude_threshold, "vec_index_magnitudes": self.vec_index_magnitudes}
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(index_parameters),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "create_index_before_load": self.create_index_before_load,
            "col_name_list": self.col_name_list,
        }

    def search_param(self) -> VexDBSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> VexDBSessionCommands:
        session_parameters = {"hnsw_ef_search": self.ef_search, "hybrid_query_ivf_probes_factor": self.hybrid_query_ivf_probes_factor}
        return {"session_options": self._optionally_build_set_options(session_parameters)}



_vexdb_case_config = {
    IndexType.ES_HNSW: VexDBHNSWConfig,
    IndexType.ES_IVFFlat: VexDBIVFFlatConfig,
    IndexType.HybridAnn: VexDBHybridANNConfig,
}
