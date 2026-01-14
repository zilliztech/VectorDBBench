import os
from pathlib import Path

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class CassandraConfig(DBConfig, BaseModel):
    """Configuration for Cassandra vector database connections.

    Supports two connection modes:
    1. Regular Cassandra: Use host and port parameters
    2. DataStax Astra DB: Use secure_connect_bundle parameter
    """
    # Regular Cassandra connection parameters
    host: str | None = None
    port: int = 9042

    # DataStax Astra DB connection (mutually exclusive with host/port)
    secure_connect_bundle: str | None = None

    # Authentication
    username: str | None = None
    password: SecretStr | None = None
    token: SecretStr | None = None  # For Astra DB token authentication

    # Keyspace
    keyspace: str = "vdb_bench"

    # Table name
    table_name: str | None = None  # Custom table name (defaults to collection_name if not specified)

    # Replication settings
    replication_strategy: str = "NetworkTopologyStrategy"  # SimpleStrategy or NetworkTopologyStrategy
    replication_factor: int = 3  # Replication factor (use 1 for single-node)
    datacenter_name: str = "datacenter1"  # Datacenter name for NetworkTopologyStrategy

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for cassandra-driver.

        Returns connection parameters formatted for Cluster initialization.
        """
        config = {}

        if self.secure_connect_bundle:
            # Resolve relative paths to absolute paths
            bundle_path = self.secure_connect_bundle
            if not os.path.isabs(bundle_path):
                # Convert relative path to absolute
                bundle_path = os.path.abspath(bundle_path)

            # Verify the bundle file exists
            if not os.path.exists(bundle_path):
                raise FileNotFoundError(
                    f"Secure connect bundle not found: {bundle_path}. "
                    f"Original path: {self.secure_connect_bundle}"
                )

            # Astra DB mode with Secure Connect Bundle
            config["cloud"] = {
                "secure_connect_bundle": bundle_path
            }
            # Astra DB uses token-based auth or username/password
            if self.token:
                config["auth_provider_token"] = self.token.get_secret_value()
            elif self.username and self.password:
                config["auth_provider_username"] = self.username
                config["auth_provider_password"] = self.password.get_secret_value()
        else:
            # Regular Cassandra mode
            config["contact_points"] = [self.host] if self.host else ["localhost"]
            config["port"] = self.port
            if self.username and self.password:
                config["auth_provider_username"] = self.username
                config["auth_provider_password"] = self.password.get_secret_value()

        config["keyspace"] = self.keyspace
        config["replication_strategy"] = self.replication_strategy
        config["replication_factor"] = self.replication_factor
        config["datacenter_name"] = self.datacenter_name
        return config


class CassandraIndexConfig(BaseModel, DBCaseConfig):
    """Index configuration for Cassandra vector search.

    Cassandra 5.0+ uses Storage-Attached Indexes (SAI) for vector search
    with support for multiple similarity functions.
    """
    metric_type: MetricType = MetricType.COSINE
    batch_size: int = 1000  # Batch size for insert operations (default: 1000)

    def parse_metric(self) -> str:
        """Map VectorDBBench metric types to Cassandra similarity functions.

        Returns:
            Cassandra similarity function name: EUCLIDEAN, DOT_PRODUCT, or COSINE
        """
        if self.metric_type == MetricType.L2:
            return "EUCLIDEAN"
        if self.metric_type == MetricType.IP:
            return "DOT_PRODUCT"
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def index_param(self) -> dict:
        """Return parameters for creating the SAI vector index.

        Returns:
            Dictionary with similarity_function and index_type
        """
        return {
            "similarity_function": self.parse_metric(),
            "index_type": "SAI"  # Storage-Attached Index
        }

    def search_param(self) -> dict:
        """Return parameters for vector search queries.

        Returns:
            Dictionary with metric for search operations
        """
        return {
            "metric": self.parse_metric()
        }
