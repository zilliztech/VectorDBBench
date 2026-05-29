from abc import abstractmethod
from typing import Any, Mapping, Optional, Sequence, TypedDict
from pydantic import BaseModel, SecretStr
from typing_extensions import LiteralString
from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

# Define the dictionary format for database connection
class GaussVectorConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
        so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str

class GaussVectorConfig(DBConfig):
    user_name: SecretStr
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 5432
    db_name: str = "postgres"

    # Convert database connection parameters to dictionary format
    def to_dict(self) -> GaussVectorConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


# GaussVector index parameters
class GaussVectorIndexParam(TypedDict):
    metric: str  # Distance metric
    index_type: str  # Index type
    index_creation_with_options: Sequence[dict[str, Any]]  # Index parameters

class GaussVectorSearchParam(TypedDict):
    metric_fun_op: LiteralString

class GaussVectorSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


# Config class for GaussVector database index-related parameters
class GaussVectorIndexConfig(BaseModel, DBCaseConfig):
    """Base Config for GaussVector"""
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    # Parse distance metric
    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "L2"
        elif self.metric_type == MetricType.COSINE:
            return "COSINE"
        return "HAMMING"

    # Parse distance metric operator (string type)
    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.L2:
            return "<->"
        elif self.metric_type == MetricType.COSINE:
            return "<+>"
        else:  # Hamming distance
            return "<#>"

    # Parse distance metric operator (function type)
    def parse_metric_fun_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.COSINE:
            return "cosine_distance"
        return "hamming_bool_distance"

    @abstractmethod
    def index_param(self) -> GaussVectorIndexParam:
        ...

    @abstractmethod
    def search_param(self) -> GaussVectorSearchParam:
        ...

    @abstractmethod
    def session_param(self) -> GaussVectorSessionCommands:
        ...

    # Convert dictionary config parameters (with_options) to structured list for building SQL WHERE clauses
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
                    }
                )
        return options

    @staticmethod
    def _optionally_build_set_options(
        set_mapping: Mapping[str, Any]
    ) -> Sequence[dict[str, Any]]:
        """Walk through options, creating 'SET 'key1 = "value1";' list"""
        session_options = []
        for setting_name, value in set_mapping.items():
            if value:
                session_options.append(
                    {"parameter": {
                            "setting_name": setting_name,
                            "val": str(value),
                        },
                    }
                )
        return session_options

class GaussVectorDiskANNConfig(GaussVectorIndexConfig, DBCaseConfig):
    queue_size: int | None = None
    index: IndexType = IndexType.GsDiskANN
    num_parallels: int | None = None
    lambda_for_balance: float | None = None
    enable_pq: bool | None = None
    subgraph_count: int | None = None
    enable_vector_copy: bool | None = None
    build_with_quantized_vector: bool | None = None
    graph_degree: int | None = None
    pq_nseg: int | None = None
    pq_nclus: int | None = None
    using_clustering_for_parallel: bool | None = None
    quantization_type: str | None = None
    lvq_nclus: int | None = None

    maintenance_work_mem: str = "8GB"  # Memory size setting
    diskann_probe_ncandidates: int | None = None
    modify_vector_index_mode: str | None = None
    version: str | None = None

    def index_param(self) -> GaussVectorIndexParam:
        # Index parameters required for index creation
        # Merge all required parameters into a dictionary
        index_parameters = {
            "queue_size": self.queue_size,
            "num_parallels": self.num_parallels,
            "lambda_for_balance": self.lambda_for_balance,
            "enable_pq": self.enable_pq,
            "subgraph_count": self.subgraph_count,
            "enable_vector_copy": self.enable_vector_copy,
            "build_with_quantized_vector": self.build_with_quantized_vector,
            "graph_degree": self.graph_degree,
            "pq_nseg": self.pq_nseg,
            "pq_nclus": self.pq_nclus,
            "using_clustering_for_parallel": self.using_clustering_for_parallel,
            "quantization_type": self.quantization_type,
            "lvq_nclus": self.lvq_nclus,
        }
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(
                index_parameters
            ),
        }

    def search_param(self) -> GaussVectorSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> GaussVectorSessionCommands:
        session_parameters = {
            "maintenance_work_mem": self.maintenance_work_mem,
            "diskann_probe_ncandidates": self.diskann_probe_ncandidates,
            "modify_vector_index_mode": self.modify_vector_index_mode,
        }
        return {
            "session_options": self._optionally_build_set_options(session_parameters)
        }


_gaussvector_case_config = {
    IndexType.GsDiskANN: GaussVectorDiskANNConfig,
}