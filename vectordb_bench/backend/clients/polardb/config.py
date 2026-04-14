from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class PolarDBConfigDict(TypedDict):
    user: str
    password: str
    host: str
    port: int
    database: str
    unix_socket: str | None = None


class PolarDBConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr | None = None
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "vectordbbench"
    unix_socket: str | None = None

    @staticmethod
    def common_long_configs() -> list[str]:
        return ["note", "unix_socket"]

    def to_dict(self) -> PolarDBConfigDict:
        pwd_str = self.password.get_secret_value() if self.password else ""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
            "database": self.database,
            "unix_socket": self.unix_socket or None,
        }


class PolarDBIndexConfig(BaseModel):
    """Base config for PolarDB vector index"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "EUCLIDEAN"
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.metric_type == MetricType.IP:
            return "INNER_PRODUCT"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)

    def parse_metric_for_distance(self) -> str:
        """Return the metric name used in DISTANCE() function"""
        if self.metric_type == MetricType.L2:
            return "EUCLIDEAN"
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.metric_type == MetricType.IP:
            return "DOT"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)


class PolarDBHNSWBaseConfig(PolarDBIndexConfig, DBCaseConfig):
    """Shared HNSW config fields for all PolarDB HNSW variants."""

    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    insert_workers: int = 10
    post_load_index: bool = False  # If True, create index after data load via ALTER TABLE
    index: IndexType = IndexType.HNSW

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric_for_distance(),
            "ef_search": self.ef_search,
        }


class PolarDBHNSWFlatConfig(PolarDBHNSWBaseConfig):
    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "metric_type_distance": self.parse_metric_for_distance(),
            "index_type": "FAISS_HNSW_FLAT",
            "M": self.M,
            "ef_construction": self.ef_construction,
            "vector_index_comment": (
                f"imci_vector_index=FAISS_HNSW_FLAT("
                f"metric={self.parse_metric()},"
                f"max_degree={self.M},"
                f"ef_construction={self.ef_construction})"
            ),
        }


class PolarDBHNSWPQConfig(PolarDBHNSWBaseConfig):
    pq_m: int = 1
    pq_nbits: int = 8

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "metric_type_distance": self.parse_metric_for_distance(),
            "index_type": "FAISS_HNSW_PQ",
            "M": self.M,
            "ef_construction": self.ef_construction,
            "vector_index_comment": (
                f"imci_vector_index=FAISS_HNSW_PQ("
                f"metric={self.parse_metric()},"
                f"max_degree={self.M},"
                f"ef_construction={self.ef_construction},"
                f"pq_m={self.pq_m},"
                f"pq_nbits={self.pq_nbits})"
            ),
        }


class PolarDBHNSWSQConfig(PolarDBHNSWBaseConfig):
    sq_type: str = "8bit"

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "metric_type_distance": self.parse_metric_for_distance(),
            "index_type": "FAISS_HNSW_SQ",
            "M": self.M,
            "ef_construction": self.ef_construction,
            "vector_index_comment": (
                f"imci_vector_index=FAISS_HNSW_SQ("
                f"metric={self.parse_metric()},"
                f"max_degree={self.M},"
                f"ef_construction={self.ef_construction},"
                f"sq_type={self.sq_type})"
            ),
        }


_polardb_case_config = {
    IndexType.HNSW: PolarDBHNSWFlatConfig,
    IndexType.HNSW_PQ: PolarDBHNSWPQConfig,
    IndexType.HNSW_SQ: PolarDBHNSWSQConfig,
}
