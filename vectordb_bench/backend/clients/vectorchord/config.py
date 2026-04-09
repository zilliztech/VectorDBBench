from abc import abstractmethod
from typing import LiteralString, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class VectorChordConfigDict(TypedDict):
    """These keys will be directly used as kwargs in psycopg connection string,
    so the names must match exactly psycopg API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class VectorChordConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str = "vectordb"

    def to_dict(self) -> VectorChordConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


_METRIC_OPS = {
    "vector": {
        MetricType.L2: "vector_l2_ops",
        MetricType.IP: "vector_ip_ops",
        MetricType.COSINE: "vector_cosine_ops",
    },
    "halfvec": {
        MetricType.L2: "halfvec_l2_ops",
        MetricType.IP: "halfvec_ip_ops",
        MetricType.COSINE: "halfvec_cosine_ops",
    },
    "rabitq8": {
        MetricType.L2: "rabitq8_l2_ops",
        MetricType.IP: "rabitq8_ip_ops",
        MetricType.COSINE: "rabitq8_cosine_ops",
    },
    "rabitq4": {
        MetricType.L2: "rabitq4_l2_ops",
        MetricType.IP: "rabitq4_ip_ops",
        MetricType.COSINE: "rabitq4_cosine_ops",
    },
}


class VectorChordIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True
    quantization_type: str = "vector"  # vector, halfvec, rabitq8, rabitq4

    def parse_metric(self) -> str:
        ops = _METRIC_OPS.get(self.quantization_type, _METRIC_OPS["vector"])
        return ops.get(self.metric_type, ops[MetricType.COSINE])

    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    @abstractmethod
    def index_param(self) -> dict: ...

    @abstractmethod
    def search_param(self) -> dict: ...

    @abstractmethod
    def session_param(self) -> dict: ...


class VectorChordRQConfig(VectorChordIndexConfig):
    index: IndexType = IndexType.VCHORDRQ
    # Build parameters (top-level options)
    residual_quantization: bool = False
    rerank_in_table: bool = False
    degree_of_parallelism: int | None = None  # default 32, range [1, 256]
    # Build parameters ([build.internal] section)
    lists: int | None = None
    spherical_centroids: bool = False
    build_threads: int | None = None  # range [1, 255]
    # PostgreSQL tuning parameter
    max_parallel_workers: int | None = None  # sets max_parallel_workers & max_parallel_maintenance_workers
    # Search parameters (GUCs)
    probes: int | None = 10
    epsilon: float | None = 1.9  # range [0.0, 4.0]
    max_scan_tuples: int | None = None  # default -1, range [-1, 2147483647]

    def index_param(self) -> dict:
        options_parts = []
        if self.rerank_in_table:
            options_parts.append("rerank_in_table = true")
        if self.residual_quantization:
            options_parts.append("residual_quantization = true")
        if self.degree_of_parallelism is not None:
            options_parts.append(f"degree_of_parallelism = {self.degree_of_parallelism}")
        options_parts.append("[build.internal]")
        if self.lists is not None:
            options_parts.append(f"lists = [{self.lists}]")
        if self.spherical_centroids:
            options_parts.append("spherical_centroids = true")
        if self.build_threads is not None:
            options_parts.append(f"build_threads = {self.build_threads}")

        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "quantization_type": self.quantization_type,
            "options": "\n".join(options_parts),
            "max_parallel_workers": self.max_parallel_workers,
        }

    def search_param(self) -> dict:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> dict:
        params = {}
        if self.probes is not None:
            params["vchordrq.probes"] = str(self.probes)
        if self.epsilon is not None:
            params["vchordrq.epsilon"] = str(self.epsilon)
        if self.max_scan_tuples is not None:
            params["vchordrq.max_scan_tuples"] = str(self.max_scan_tuples)
        return params


class VectorChordGraphConfig(VectorChordIndexConfig):
    index: IndexType = IndexType.VCHORDG
    # Build parameters
    m: int | None = None  # default 32, max neighbors per vertex
    ef_construction: int | None = None  # default 64
    bits: int | None = None  # default 2, quantization ratio (1 or 2)
    # PostgreSQL tuning parameter
    max_parallel_workers: int | None = None
    # Search parameters (GUCs)
    ef_search: int | None = 64  # range [1, 65535]
    beam_search: int | None = None  # default 1
    max_scan_tuples: int | None = None  # default -1, range [-1, 2147483647]

    def index_param(self) -> dict:
        options_parts = []
        if self.m is not None:
            options_parts.append(f"m = {self.m}")
        if self.ef_construction is not None:
            options_parts.append(f"ef_construction = {self.ef_construction}")
        if self.bits is not None:
            options_parts.append(f"bits = {self.bits}")

        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "quantization_type": self.quantization_type,
            "options": "\n".join(options_parts),
            "max_parallel_workers": self.max_parallel_workers,
        }

    def search_param(self) -> dict:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> dict:
        params = {}
        if self.ef_search is not None:
            params["vchordg.ef_search"] = str(self.ef_search)
        if self.beam_search is not None:
            params["vchordg.beam_search"] = str(self.beam_search)
        if self.max_scan_tuples is not None:
            params["vchordg.max_scan_tuples"] = str(self.max_scan_tuples)
        return params


_vectorchord_case_config = {
    IndexType.VCHORDRQ: VectorChordRQConfig,
    IndexType.VCHORDG: VectorChordGraphConfig,
}
