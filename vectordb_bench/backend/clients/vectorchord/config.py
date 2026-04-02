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
    user_name: SecretStr = SecretStr("postgres")
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


class VectorChordIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

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


_vectorchord_case_config = {
    IndexType.VCHORDRQ: VectorChordRQConfig,
}
