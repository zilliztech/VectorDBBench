from abc import abstractmethod
from typing import Any, Mapping, Optional, Sequence, TypedDict
from pydantic import BaseModel, SecretStr
from typing_extensions import LiteralString
from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgVectorConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
        so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class PgVectorConfig(DBConfig):
    user_name: SecretStr = SecretStr("postgres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> PgVectorConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class PgVectorIndexParam(TypedDict):
    metric: str
    index_type: str
    index_creation_with_options: Sequence[dict[str, Any]]
    maintenance_work_mem: Optional[str]
    max_parallel_workers: Optional[int]


class PgVectorSearchParam(TypedDict):
    metric_fun_op: LiteralString


class PgVectorSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class PgVectorIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    def parse_metric(self) -> str:
        if self.quantization_type == "halfvec":
            if self.metric_type == MetricType.L2:
                return "halfvec_l2_ops"
            elif self.metric_type == MetricType.IP:
                return "halfvec_ip_ops"
            return "halfvec_cosine_ops"
        elif self.quantization_type == "bit":
            if self.metric_type == MetricType.JACCARD:
                return "bit_jaccard_ops"
            return "bit_hamming_ops"
        else:
            if self.metric_type == MetricType.L2:
                return "vector_l2_ops"
            elif self.metric_type == MetricType.IP:
                return "vector_ip_ops"
            return "vector_cosine_ops"

    def parse_metric_fun_op(self) -> LiteralString:
        if self.quantization_type == "bit":
            if self.metric_type == MetricType.JACCARD:
                return "<%>"
            return "<~>"
        else:
            if self.metric_type == MetricType.L2:
                return "<->"
            elif self.metric_type == MetricType.IP:
                return "<#>"
            return "<=>"

    def parse_metric_fun_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"
    
    def parse_reranking_metric_fun_op(self) -> LiteralString:
        if self.reranking_metric == MetricType.L2:
            return "<->"
        elif self.reranking_metric == MetricType.IP:
            return "<#>"
        return "<=>"


    @abstractmethod
    def index_param(self) -> PgVectorIndexParam:
        ...

    @abstractmethod
    def search_param(self) -> PgVectorSearchParam:
        ...

    @abstractmethod
    def session_param(self) -> PgVectorSessionCommands:
        ...

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


class PgVectorIVFFlatConfig(PgVectorIndexConfig):
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
    maintenance_work_mem: Optional[str] = None
    max_parallel_workers: Optional[int] = None
    quantization_type: Optional[str] = None
    table_quantization_type: Optional[str] = None
    reranking: Optional[bool] = None
    quantized_fetch_limit: Optional[int] = None
    reranking_metric: Optional[str] = None

    def index_param(self) -> PgVectorIndexParam:
        index_parameters = {"lists": self.lists}
        if self.quantization_type == "none" or self.quantization_type == None:
            self.quantization_type = "vector"
        if self.table_quantization_type == "none" or self.table_quantization_type == None:
            self.table_quantization_type = "vector"
        if self.table_quantization_type == "bit":
            self.quantization_type = "bit"
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(
                index_parameters
            ),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "quantization_type": self.quantization_type,
            "table_quantization_type": self.table_quantization_type,
        }

    def search_param(self) -> PgVectorSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
            "reranking": self.reranking,
            "reranking_metric_fun_op": self.parse_reranking_metric_fun_op(),
            "quantized_fetch_limit": self.quantized_fetch_limit,
        }

    def session_param(self) -> PgVectorSessionCommands:
        session_parameters = {"ivfflat.probes": self.probes}
        return {
            "session_options": self._optionally_build_set_options(session_parameters)
        }


class PgVectorHNSWConfig(PgVectorIndexConfig):
    """
    An HNSW index creates a multilayer graph. It has better query performance than IVFFlat (in terms of
    speed-recall tradeoff), but has slower build times and uses more memory. Also, an index can be
    created without any data in the table since there isn't a training step like IVFFlat.
    """

    m: int | None  # DETAIL:  Valid values are between "2" and "100".
    ef_construction: (
        int | None
    )  # ef_construction must be greater than or equal to 2 * m
    ef_search: int | None
    index: IndexType = IndexType.ES_HNSW
    maintenance_work_mem: Optional[str] = None
    max_parallel_workers: Optional[int] = None
    quantization_type: Optional[str] = None
    table_quantization_type: Optional[str] = None
    reranking: Optional[bool] = None
    quantized_fetch_limit: Optional[int] = None
    reranking_metric: Optional[str] = None

    def index_param(self) -> PgVectorIndexParam:
        index_parameters = {"m": self.m, "ef_construction": self.ef_construction}
        if self.quantization_type == "none" or self.quantization_type == None:
            self.quantization_type = "vector"
        if self.table_quantization_type == "none" or self.table_quantization_type == None:
            self.table_quantization_type = "vector"
        if self.table_quantization_type == "bit":
            self.quantization_type = "bit"
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(
                index_parameters
            ),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "quantization_type": self.quantization_type,
            "table_quantization_type": self.table_quantization_type,
        }

    def search_param(self) -> PgVectorSearchParam:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
            "reranking": self.reranking,
            "reranking_metric_fun_op": self.parse_reranking_metric_fun_op(),
            "quantized_fetch_limit": self.quantized_fetch_limit,
        }

    def session_param(self) -> PgVectorSessionCommands:
        session_parameters = {"hnsw.ef_search": self.ef_search}
        return {
            "session_options": self._optionally_build_set_options(session_parameters)
        }


_pgvector_case_config = {
        IndexType.HNSW: PgVectorHNSWConfig,
        IndexType.ES_HNSW: PgVectorHNSWConfig,
        IndexType.IVFFlat: PgVectorIVFFlatConfig,
}
