"""Configuration classes for CockroachDB vector database integration."""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, LiteralString, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class CockroachDBConfigDict(TypedDict):
    """Connection configuration for CockroachDB using psycopg."""

    user: str
    password: str
    host: str
    port: int
    dbname: str
    sslmode: str


class CockroachDBConfig(DBConfig):
    """Main configuration for CockroachDB connection."""

    user_name: SecretStr = "root"
    password: SecretStr | None = None
    host: str = "localhost"
    port: int = 26257
    db_name: str = "defaultdb"
    table_name: str = "vdbbench_cockroachdb"
    isolation_level: str = "serializable"
    pool_size: int = 100
    max_overflow: int = 100
    pool_recycle: int = 3600
    connect_timeout: int = 10
    sslmode: str = "disable"  # Options: disable, require, verify-ca, verify-full
    sslrootcert: str | None = None  # Path to CA cert (for verify-ca, verify-full)
    sslcert: str | None = None  # Path to client cert (for mutual TLS)
    sslkey: str | None = None  # Path to client key (for mutual TLS)

    def to_dict(self) -> CockroachDBConfigDict:
        user_str = self.user_name.get_secret_value() if isinstance(self.user_name, SecretStr) else self.user_name
        pwd_str = self.password.get_secret_value() if self.password else ""

        connect_config = {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
            "sslmode": self.sslmode,
        }

        # Add SSL certificate paths if provided
        if self.sslrootcert:
            connect_config["sslrootcert"] = self.sslrootcert
        if self.sslcert:
            connect_config["sslcert"] = self.sslcert
        if self.sslkey:
            connect_config["sslkey"] = self.sslkey

        return {
            "connect_config": connect_config,
            "table_name": self.table_name,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_recycle": self.pool_recycle,
            "connect_timeout": self.connect_timeout,
        }


class CockroachDBIndexParam(TypedDict):
    """Index parameters for CockroachDB vector indexes."""

    metric: str
    index_creation_with_options: Sequence[dict[str, Any]]
    min_partition_size: int | None
    max_partition_size: int | None
    build_beam_size: int | None


class CockroachDBSearchParam(TypedDict):
    """Search parameters for CockroachDB vector queries."""

    metric_fun_op: LiteralString
    vector_search_beam_size: int | None


class CockroachDBSessionCommands(TypedDict):
    """Session-level commands for CockroachDB."""

    session_options: Sequence[dict[str, Any]]


class CockroachDBIndexConfig(BaseModel, DBCaseConfig):
    """Base configuration for CockroachDB vector indexes."""

    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True
    min_partition_size: int | None = 16
    max_partition_size: int | None = 128
    build_beam_size: int | None = 8
    vector_search_beam_size: int | None = 32

    def parse_metric(self) -> str:
        """Parse metric type to CockroachDB opclass."""
        metric_map = {
            MetricType.L2: "vector_l2_ops",
            MetricType.IP: "vector_ip_ops",
            MetricType.COSINE: "vector_cosine_ops",
        }
        metric = metric_map.get(self.metric_type)
        if metric is None:
            msg = f"Unsupported metric type: {self.metric_type}"
            raise ValueError(msg)
        return metric

    def parse_metric_fun_op(self) -> LiteralString:
        """Parse metric type to distance operator."""
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    @abstractmethod
    def index_param(self) -> CockroachDBIndexParam: ...

    @abstractmethod
    def search_param(self) -> CockroachDBSearchParam: ...

    @abstractmethod
    def session_param(self) -> CockroachDBSessionCommands: ...

    @staticmethod
    def _optionally_build_with_options(with_options: Mapping[str, Any]) -> Sequence[dict[str, Any]]:
        """Build WITH options for index creation."""
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
        """Build SET options for session configuration."""
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


class CockroachDBVectorIndexConfig(CockroachDBIndexConfig):
    """
    CockroachDB Vector Index Configuration using C-SPANN algorithm.

    Available since CockroachDB v25.2. Uses hierarchical k-means clustering
    for efficient approximate nearest neighbor (ANN) search.

    Tunable parameters:
    - min_partition_size: Minimum vectors per partition (default: 16, range: 1-1024)
    - max_partition_size: Maximum vectors per partition (default: 128, range: 4x min-4096)
    - vector_search_beam_size: Partitions explored during search (default: 32)
    """

    index: IndexType = IndexType.Flat

    def index_param(self) -> CockroachDBIndexParam:
        """Get index creation parameters."""
        index_parameters = {
            "min_partition_size": self.min_partition_size,
            "max_partition_size": self.max_partition_size,
            "build_beam_size": self.build_beam_size,
        }

        return {
            "metric": self.parse_metric(),
            "index_creation_with_options": self._optionally_build_with_options(index_parameters),
            "min_partition_size": self.min_partition_size,
            "max_partition_size": self.max_partition_size,
            "build_beam_size": self.build_beam_size,
        }

    def search_param(self) -> CockroachDBSearchParam:
        """Get search parameters."""
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
            "vector_search_beam_size": self.vector_search_beam_size,
        }

    def session_param(self) -> CockroachDBSessionCommands:
        """Get session parameters."""
        session_parameters = {"vector_search_beam_size": self.vector_search_beam_size}
        return {"session_options": self._optionally_build_set_options(session_parameters)}


_cockroachdb_case_config = {
    IndexType.Flat: CockroachDBVectorIndexConfig,
    IndexType.HNSW: CockroachDBVectorIndexConfig,
    IndexType.IVFFlat: CockroachDBVectorIndexConfig,
}
