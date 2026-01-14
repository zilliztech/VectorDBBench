import logging
from contextlib import contextmanager

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, BatchType

from ..api import VectorDB
from .config import CassandraIndexConfig

log = logging.getLogger(__name__)


class Cassandra(VectorDB):
    """Cassandra vector database client.

    Supports both regular Cassandra (5.0+) and DataStax Astra DB
    with vector search capabilities using Storage-Attached Indexes (SAI).
    """

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: CassandraIndexConfig,
        collection_name: str = "vdb_bench_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize Cassandra client.

        Args:
            dim: Vector dimension
            db_config: Database configuration dictionary from CassandraConfig.to_dict()
            db_case_config: Index configuration
            collection_name: Table name for vector storage
            drop_old: Whether to drop existing table
        """
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.keyspace = db_config["keyspace"]

        # Field names
        self.id_field = "id"
        self.vector_field = "vector"

        # Initialize connection to setup keyspace/table and drop if needed
        cluster, session = self._create_cluster_and_session()

        # Create keyspace if not exists (must be done before dropping/creating tables)
        self._create_keyspace(session)

        if drop_old:
            log.info(f"Dropping old table: {self.keyspace}.{self.table_name}")
            session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.table_name}")

        # Create table
        self._create_table(session, dim)

        # Create index immediately after table creation
        self._create_index(session)

        # Close initial connection
        cluster.shutdown()
        self.cluster = None
        self.session = None

    def _create_cluster_and_session(self):
        """Create Cassandra cluster and session based on configuration.

        Returns:
            Tuple of (Cluster, Session)
        """
        config = self.db_config

        if "cloud" in config:
            # Astra DB with Secure Connect Bundle
            cloud_config = config["cloud"]

            # Setup authentication
            if "auth_provider_token" in config:
                auth_provider = PlainTextAuthProvider("token", config["auth_provider_token"])
            elif "auth_provider_username" in config:
                auth_provider = PlainTextAuthProvider(
                    config["auth_provider_username"],
                    config["auth_provider_password"]
                )
            else:
                auth_provider = None

            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        else:
            # Regular Cassandra
            contact_points = config.get("contact_points", ["localhost"])
            port = config.get("port", 9042)

            if "auth_provider_username" in config:
                auth_provider = PlainTextAuthProvider(
                    config["auth_provider_username"],
                    config["auth_provider_password"]
                )
            else:
                auth_provider = None

            cluster = Cluster(contact_points=contact_points, port=port, auth_provider=auth_provider)

        session = cluster.connect()
        return cluster, session

    def _create_keyspace(self, session):
        """Create keyspace if it doesn't exist.

        Args:
            session: Cassandra session
        """
        # First try to use the keyspace if it already exists
        try:
            session.set_keyspace(self.keyspace)
            log.info(f"Using existing keyspace: {self.keyspace}")
            return
        except Exception:
            # Keyspace doesn't exist, try to create it
            log.info(f"Keyspace {self.keyspace} does not exist, attempting to create it")

        # Try to create the keyspace
        try:
            replication_strategy = self.db_config.get("replication_strategy", "NetworkTopologyStrategy")
            replication_factor = self.db_config.get("replication_factor", 3)
            datacenter_name = self.db_config.get("datacenter_name", "datacenter1")
            
            # Build replication settings based on strategy
            if replication_strategy == "NetworkTopologyStrategy":
                replication_settings = f"{{'class': '{replication_strategy}', '{datacenter_name}': {replication_factor}}}"
            else:
                replication_settings = f"{{'class': '{replication_strategy}', 'replication_factor': {replication_factor}}}"
            
            cql = f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH REPLICATION = {replication_settings}
            """
            print(cql)
            session.execute(cql)
            session.set_keyspace(self.keyspace)
            log.info(f"Created and using keyspace: {self.keyspace} with replication strategy: {replication_strategy}, datacenter: {datacenter_name}, factor: {replication_factor}")
        except Exception as e:
            log.error(f"Failed to create keyspace {self.keyspace}: {e}")
            # Try to use it anyway in case it was created by another process
            try:
                session.set_keyspace(self.keyspace)
                log.info(f"Using keyspace: {self.keyspace}")
            except Exception as e2:
                log.error(f"Failed to use keyspace {self.keyspace}: {e2}")
                raise

    def _create_table(self, session, dim: int):
        """Create table with vector column.

        Args:
            session: Cassandra session
            dim: Vector dimension
        """
        cql = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table_name} (
            {self.id_field} bigint PRIMARY KEY,
            {self.vector_field} VECTOR<FLOAT, {dim}>
        )
        """
        session.execute(cql)
        log.info(f"Created table {self.keyspace}.{self.table_name} with vector dimension {dim}")

    def _create_index(self, session):
        """Create SAI vector index for optimized vector search.

        Args:
            session: Cassandra session
        """
        index_name = f"{self.table_name}_vector_idx"
        index_params = self.case_config.index_param()
        similarity_function = index_params["similarity_function"]

        # Drop existing index if present
        cql_drop = f"DROP INDEX IF EXISTS {self.keyspace}.{index_name}"
        session.execute(cql_drop)
        log.info(f"Dropped existing index {index_name} if present")

        # Create SAI vector index
        cql = f"""
        CREATE CUSTOM INDEX {index_name}
        ON {self.keyspace}.{self.table_name} ({self.vector_field})
        USING 'StorageAttachedIndex'
        WITH OPTIONS = {{'similarity_function': '{similarity_function}'}}
        """
        session.execute(cql)
        log.info(f"Created vector index {index_name} with similarity function {similarity_function}")

    @contextmanager
    def init(self):
        """Initialize Cassandra client and cleanup when done.

        Yields control to execute operations within the context.
        """
        try:
            log.debug("Initializing Cassandra connection")
            self.cluster, self.session = self._create_cluster_and_session()
            self.session.set_keyspace(self.keyspace)
            log.debug(f"Successfully connected to keyspace: {self.keyspace}")
            yield
        except Exception as e:
            log.error(f"Failed to initialize Cassandra connection: {e}")
            raise
        finally:
            if self.cluster is not None:
                log.debug("Shutting down Cassandra connection")
                self.cluster.shutdown()
                self.cluster = None
                self.session = None

    def need_normalize_cosine(self) -> bool:
        """Whether database requires normalized vectors for cosine similarity.

        Cassandra handles cosine normalization internally.

        Returns:
            False - Cassandra handles normalization
        """
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings using batch statements.

        Args:
            embeddings: List of vector embeddings
            metadata: List of IDs for each embedding
            **kwargs: Additional parameters (unused)

        Returns:
            Tuple of (count of inserted records, exception if any)
        """
        if self.session is None:
            log.error("Cannot insert: session is None. Make sure insert is called within init() context manager.")
            return 0, RuntimeError("Session not initialized")

        try:
            # Cassandra batch statements have size limits, so batch in chunks
            batch_size = self.case_config.batch_size
            total_inserted = 0

            insert_cql = f"""
            INSERT INTO {self.keyspace}.{self.table_name} ({self.id_field}, {self.vector_field})
            VALUES (?, ?)
            """
            prepared = self.session.prepare(insert_cql)

            for i in range(0, len(embeddings), batch_size):
                batch = BatchStatement(batch_type=BatchType.UNLOGGED)
                end_idx = min(i + batch_size, len(embeddings))

                for id_, embedding in zip(metadata[i:end_idx], embeddings[i:end_idx], strict=False):
                    batch.add(prepared, (id_, embedding))

                self.session.execute(batch)
                total_inserted += (end_idx - i)

                if (i // batch_size) % 10 == 0 and i > 0:
                    log.debug(f"Inserted {total_inserted} embeddings so far...")

            log.info(f"Successfully inserted {total_inserted} embeddings")
            return total_inserted, None
        except Exception as e:
            log.error(f"Error inserting embeddings: {e}")
            return total_inserted if 'total_inserted' in locals() else 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        """Search for similar vectors using ANN (Approximate Nearest Neighbor).

        Args:
            query: Query vector
            k: Number of results to return
            filters: Optional filters (not implemented for Cassandra)
            **kwargs: Additional parameters (unused)

        Returns:
            List of IDs ordered by similarity (most similar first)
        """
        if self.session is None:
            log.error("Cannot search: session is None. Make sure search is called within init() context manager.")
            raise RuntimeError("Session not initialized. Call search within init() context manager.")

        try:
            # Cassandra uses ANN OF for vector similarity search
            # The similarity function is determined by the index
            cql = f"""
            SELECT {self.id_field}
            FROM {self.keyspace}.{self.table_name}
            ORDER BY {self.vector_field} ANN OF %s
            LIMIT %s
            """

            results = self.session.execute(cql, (query, k))
            result_list = [row[0] for row in results]
            log.debug(f"Search returned {len(result_list)} results")
            return result_list
        except Exception as e:
            log.error(f"Search query failed: {e}. This usually indicates the vector index hasn't been created. Query: {cql[:100]}")
            raise

    def optimize(self, data_size: int | None = None) -> None:
        """Optimize operation - no action needed since index is created during initialization.

        The index is now created immediately after table creation in __init__,
        before any data is inserted. This method is kept for API compatibility.

        Args:
            data_size: Size of data (unused, kept for API compatibility)
        """
        log.info("Index already created during initialization - no optimization needed")
        pass

    def ready_to_load(self) -> None:
        """Prepare for data loading.

        Cassandra is always ready to load data.
        """
        pass
