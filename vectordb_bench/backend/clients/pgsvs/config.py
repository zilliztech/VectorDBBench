from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, LiteralString, TypedDict

from pydantic import BaseModel, SecretStr, root_validator, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgSVSConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class PgSVSConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str = "vectordb"
    table_name: str = "vdbbench_table_test"

    def to_dict(self) -> PgSVSConfigDict:
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


class PgSVSIndexParam(TypedDict):
    metric: str
    index_type: str
    index_creation_with_options: Sequence[dict[str, Any]]
    maintenance_work_mem: str | None
    max_parallel_workers: int | None
    quantization_type: str | None
    table_quantization_type: str | None


class PgSVSSearchParam(TypedDict):
    metric_fun_op: LiteralString
    reranking: bool | None
    reranking_metric_fun_op: LiteralString
    quantized_fetch_limit: int | None
    search_window_size: int | None
    use_search_history: bool | None


class PgSVSSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class PgSVSIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True
    # Scan more of the index to get enough results for filter-cases.
    # Options: "strict_order" (order by distance), "relaxed_order" (slightly out of order but better recall)
    # See: https://github.com/pgvector/pgvector?tab=readme-ov-file#iterative-index-scans
    iterative_scan: str = "relaxed_order"

    def parse_metric(self, metric_type: MetricType | None = None) -> str:
        effective_metric_type = self.metric_type if metric_type is None else metric_type
        d = {
            "halfvec": {
                MetricType.L2: "halfvec_l2_ops",
                MetricType.IP: "halfvec_ip_ops",
                MetricType.COSINE: "halfvec_cosine_ops",
            },
            "bit": {
                MetricType.JACCARD: "bit_jaccard_ops",
                MetricType.HAMMING: "bit_hamming_ops",
            },
            "_fallback": {
                MetricType.L2: "vector_l2_ops",
                MetricType.IP: "vector_ip_ops",
                MetricType.COSINE: "vector_cosine_ops",
            },
        }

        if d.get(self.quantization_type) is None:
            return d.get("_fallback").get(effective_metric_type)
        metric = d.get(self.quantization_type).get(effective_metric_type)
        # If using binary quantization for the index, use a bit metric
        # no matter what metric was selected for vector or halfvec data
        if self.quantization_type == "bit" and metric is None:
            return "bit_hamming_ops"
        return metric

    def parse_metric_fun_op(self, metric_type: MetricType | None = None) -> LiteralString:
        effective_metric_type = self.metric_type if metric_type is None else metric_type
        if self.quantization_type == "bit":
            if effective_metric_type == MetricType.JACCARD:
                return "<%>"
            return "<~>"
        if effective_metric_type == MetricType.L2:
            return "<->"
        if effective_metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    def parse_metric_fun_str(self, metric_type: MetricType | None = None) -> str:
        effective_metric_type = self.metric_type if metric_type is None else metric_type
        if effective_metric_type == MetricType.L2:
            return "l2_distance"
        if effective_metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"

    def parse_reranking_metric_fun_op(self) -> LiteralString:
        if self.reranking_metric == MetricType.L2:
            return "<->"
        if self.reranking_metric == MetricType.IP:
            return "<#>"
        return "<=>"

    @abstractmethod
    def index_param(self) -> PgSVSIndexParam: ...

    @abstractmethod
    def search_param(self) -> PgSVSSearchParam: ...

    @abstractmethod
    def session_param(self) -> PgSVSSessionCommands: ...

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


class PgSVSVamanaConfig(PgSVSIndexConfig):
    compression_type: int | None = None
    compression_primary: int | None = None
    compression_secondary: int | None = None
    alpha: int | None = None
    build_window_size: int | None
    search_window_size: int | None
    use_search_history: bool | None = None
    graph_degree: int | None = None
    leanvec_dims: int | None = None
    index: IndexType = IndexType.SVS_VAMANA
    quantization_type: str | None = None
    table_quantization_type: str | None
    maintenance_work_mem: str | None = None
    max_parallel_workers: int | None = None
    reranking: bool | None = None
    quantized_fetch_limit: int | None = None
    reranking_metric: str | None = None
    normalize_cosine: bool = False

    @validator("quantization_type", "table_quantization_type")
    def reject_bit_quantization(cls, value: str | None, field: Any):
        # vamana has no bit opclass; CREATE INDEX would fail only after the full data load
        if value == "bit":
            msg = f"{field.name}=bit is not supported for svs (vamana has no bit operator class); use none or halfvec"
            raise ValueError(msg)
        return value

    @root_validator(skip_on_failure=True)
    def reject_reranking(cls, values: dict):
        # pgvector-level reranking only applies to bit indexes; svs reranks internally via compression_secondary
        if values.get("reranking"):
            msg = "reranking is not supported for svs; svs reranks internally via compression_secondary"
            raise ValueError(msg)
        if values.get("quantized_fetch_limit") is not None:
            msg = "quantized_fetch_limit is not supported for svs (it only applies to bit quantization with reranking)"
            raise ValueError(msg)
        return values

    @validator("graph_degree")
    def validate_graph_degree(cls, value: int | None):
        # matches the vamana_relopt_kind "graph_degree" reloption bound (vamana.h VAMANA_MIN/MAX_GRAPH_DEGREE)
        if value is not None and not (16 <= value <= 256):
            msg = f"graph_degree={value} is out of range for svs; valid range is 16-256"
            raise ValueError(msg)
        return value

    @validator("alpha")
    def validate_alpha(cls, value: int | None):
        # matches the "alpha" reloption bound; -1 or 0 both mean "use the SVS library default"
        if value is not None and not (-1 <= value <= 200):
            msg = f"alpha={value} is out of range for svs; valid values are -1 (library default) or 0-200 (alpha*100)"
            raise ValueError(msg)
        return value

    @validator("build_window_size")
    def validate_build_window_size(cls, value: int | None):
        # matches the "build_window_size" reloption bound
        if value is not None and not (-1 <= value <= 1000):
            msg = f"build_window_size={value} is out of range for svs; valid values are -1 (2 * graph_degree) or 0-1000"
            raise ValueError(msg)
        return value

    @validator("search_window_size")
    def validate_search_window_size(cls, value: int | None):
        # matches the "search_window_size" reloption/GUC bound; unlike build_window_size there is no -1
        # sentinel here despite the reloption's description string claiming one -- Postgres's own bound
        # check (VAMANA_MIN_SEARCH_WINDOW=10) would reject -1 before it ever reached that fallback.
        if value is not None and not (10 <= value <= 10000):
            msg = f"search_window_size={value} is out of range for svs; valid range is 10-10000"
            raise ValueError(msg)
        return value

    @validator("compression_type")
    def validate_compression_type(cls, value: int | None):
        if value is not None and value not in (0, 1, 2):
            msg = f"compression_type={value} is invalid for svs; valid values are 0 (none), 1 (leanvec), 2 (lvq)"
            raise ValueError(msg)
        return value

    @validator("compression_primary")
    def validate_compression_primary(cls, value: int | None):
        # svs only accepts these four discrete codes (MapCompressionParamToSVSType in svs_wrapper.c);
        # the reloption's numeric bound (-8 to 8) is wider than what's actually accepted
        if value is not None and value not in (-8, -4, 4, 8):
            msg = f"compression_primary={value} is invalid for svs; valid values are -8, -4, 4, 8"
            raise ValueError(msg)
        return value

    @validator("leanvec_dims")
    def validate_leanvec_dims(cls, value: int | None):
        # matches the "leanvec_dims" reloption bound
        if value is not None and not (-1 <= value <= 2000):
            msg = f"leanvec_dims={value} is out of range for svs; valid values are -1 (dimensions/2) or 0-2000"
            raise ValueError(msg)
        return value

    @root_validator(skip_on_failure=True)
    def validate_compression_secondary(cls, values: dict):
        secondary = values.get("compression_secondary")
        if secondary is None:
            return values
        # 0 ("no residual") is only accepted for LVQ's residual slot, not LeanVec's secondary encoding
        if secondary == 0 and values.get("compression_type") == 2:
            return values
        if secondary not in (-8, -4, 4, 8):
            msg = (
                f"compression_secondary={secondary} is invalid for svs; valid values are -8, -4, 4, 8"
                " (or 0 for no residual, only when compression_type=2/lvq)"
            )
            raise ValueError(msg)
        return values

    def _effective_metric_type(self) -> MetricType | None:
        if self.normalize_cosine and self.metric_type == MetricType.COSINE:
            return MetricType.L2
        return self.metric_type

    def index_param(self) -> PgSVSIndexParam:
        index_parameters = {
            "alpha": self.alpha,
            "graph_degree": self.graph_degree,
            "leanvec_dims": self.leanvec_dims,
            "build_window_size": self.build_window_size,
            "use_search_history": self.use_search_history,
            "compression_type": self.compression_type,
            "compression_primary": self.compression_primary,
            "compression_secondary": self.compression_secondary,
        }
        if self.quantization_type == "none" or self.quantization_type is None:
            self.quantization_type = "vector"
        if self.table_quantization_type == "none" or self.table_quantization_type is None:
            self.table_quantization_type = "vector"
        if self.table_quantization_type == "bit":
            self.quantization_type = "bit"
        effective_metric_type = self._effective_metric_type()
        return {
            "metric": self.parse_metric(metric_type=effective_metric_type),
            "index_type": self.index.value,
            "index_creation_with_options": self._optionally_build_with_options(index_parameters),
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_parallel_workers": self.max_parallel_workers,
            "quantization_type": self.quantization_type,
            "table_quantization_type": self.table_quantization_type,
        }

    def search_param(self) -> PgSVSSearchParam:
        effective_metric_type = self._effective_metric_type()
        return {
            "metric_fun_op": self.parse_metric_fun_op(metric_type=effective_metric_type),
            "reranking": self.reranking,
            "reranking_metric_fun_op": self.parse_reranking_metric_fun_op(),
            "quantized_fetch_limit": self.quantized_fetch_limit,
            "search_window_size": self.search_window_size,
            "use_search_history": self.use_search_history,
        }

    def session_param(self) -> PgSVSSessionCommands:
        session_parameters = {"svs.search_window_size": self.search_window_size}
        return {"session_options": self._optionally_build_set_options(session_parameters)}


_pgsvs_case_config = {
    IndexType.SVS_VAMANA: PgSVSVamanaConfig,
}
