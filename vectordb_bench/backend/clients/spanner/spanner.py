import time
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple

from google.cloud import spanner_v1
from google.auth.exceptions import DefaultCredentialsError
from vectordb_bench.backend.clients.api import VectorDB, MetricType, DBCaseConfig
from vectordb_bench.backend.clients.spanner.config import SpannerConfig, SpannerCaseConfig
from vectordb_bench.backend.dataset import Dataset

# Attempt to set credentials from environment variable if not already set
# This is useful for local development or CI environments
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and "SPANNER_EMULATOR_HOST" not in os.environ:
    # Replace with the actual path to your service account key file if needed
    # or ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment.
    # For emulator, SPANNER_EMULATOR_HOST should be set.
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set and SPANNER_EMULATOR_HOST not set.")
    print("Spanner client may not be able to authenticate unless running on GCP with default credentials.")

class SpannerVectorDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str, # This will be used as the table name
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = SpannerConfig(**db_config)
        self.case_config: SpannerCaseConfig = db_case_config # Explicitly type hint
        self.table_name = self.db_config.table_name # Use table_name from SpannerConfig
        self.project_id = self.db_config.project_id
        self.instance_id = self.db_config.instance_id
        self.database_id = self.db_config.database_id
        self.emulator_host = self.db_config.emulator_host

        self.spanner_client = None
        self.instance = None
        self.database = None

        if self.emulator_host:
            print(f"Using Spanner Emulator: {self.emulator_host}")
            self.spanner_client = spanner_v1.Client(project=self.project_id, client_options={"api_endpoint": self.emulator_host})
        else:
            try:
                self.spanner_client = spanner_v1.Client(project=self.project_id)
            except DefaultCredentialsError:
                print("ERROR: Could not automatically determine credentials. " \
                      "Please set GOOGLE_APPLICATION_CREDENTIALS or ensure you are running " \
                      "in an environment with default credentials (e.g., a GCE instance).")
                raise

        self.instance = self.spanner_client.instance(self.instance_id)
        self.database = self.instance.database(self.database_id)

        if not self.instance.exists():
            raise RuntimeError(f"Spanner instance {self.instance_id} does not exist in project {self.project_id}.")

        if not self.database.exists():
            print(f"Database {self.database_id} does not exist. Creating database...")
            # DDL for database creation (if needed, though typically pre-exists)
            # For this benchmark, we assume the database exists or is created outside.
            # If creation is required, it would involve specific DDL commands.
            # self.database.create().result() # This is a long-running operation
            raise RuntimeError(f"Spanner database {self.database_id} does not exist. Please create it first.")


        if drop_old:
            self._drop_table()

        self._create_table_if_not_exists()

    @contextmanager
    def init(self) -> None:
        # Spanner client is initialized in __init__
        # Connection pooling is handled by the client library.
        # For this context manager, we don't need to do anything specific for setup/teardown per call.
        # If there were per-call resource acquisition/release, it would go here.
        try:
            yield
        finally:
            # Client is long-lived, no specific close needed here for each operation set.
            # If we wanted to close the client after all operations, it would be outside this init.
            pass

    def _table_exists(self) -> bool:
        """Checks if the table already exists in the database."""
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                f"SELECT table_name FROM information_schema.tables WHERE table_name = '{self.table_name}'"
            )
            return any(results)

    def _create_table_if_not_exists(self):
        if not self._table_exists():
            print(f"Table {self.table_name} does not exist. Creating table with FLOAT32 embedding column...")
            # DDL for table creation with embedding column as ARRAY<FLOAT32>(vector_length=dim) NOT NULL
            table_ddl = f"""CREATE TABLE {self.table_name} (
                id INT64 NOT NULL,
                embedding ARRAY<FLOAT32>(vector_length={self.dim}) NOT NULL
            ) PRIMARY KEY (id)"""

            ddl_statements = [table_ddl]

            print(f"Attempting to execute DDL for table creation: {table_ddl}")

            operation = self.database.update_ddl(ddl_statements)
            operation.result() # Wait for completion
            print(f"Table {self.table_name} creation process completed.")
        else:
            print(f"Table {self.table_name} already exists.")


    def _drop_table(self):
        if self._table_exists():
            print(f"Dropping table {self.table_name}...")
            ddl_statements = [f"DROP TABLE {self.table_name}"]
            operation = self.database.update_ddl(ddl_statements)
            operation.result() # Wait for completion
            print(f"Table {self.table_name} dropped successfully.")

    def insert_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[int],
        **kwargs,
    ) -> Tuple[int, Optional[Exception]]:
        if not embeddings:
            return 0, None

        rows = [(metadata[i], embedding) for i, embedding in enumerate(embeddings)]

        # Define columns for insert
        columns = ['id', 'embedding']

        try:
            with self.database.batch() as batch:
                batch.insert(
                    table=self.table_name,
                    columns=columns,
                    values=rows,
                )
            # print(f"Successfully inserted {len(rows)} rows into {self.table_name}.")
            return len(rows), None
        except Exception as e:
            print(f"Error inserting embeddings: {e}")
            return 0, e

    def search_embedding(
        self,
        query: List[float],
        k: int = 100,
        filters: Optional[dict] = None,
    ) -> List[int]:
        results_list = []
        num_leaves_to_search_param = self.case_config.search_param()['num_leaves_to_search']

        # Using APPROX_COSINE_DISTANCE with JSON_OBJECT options
        sql_query = f"""
        SELECT id
        FROM {self.table_name}
        ORDER BY APPROX_COSINE_DISTANCE(embedding, @query_vector, options => JSON_OBJECT('num_leaves_to_search', @num_leaves_to_search_param))
        LIMIT @k
        """

        params = {
            "query_vector": query, # Python list of floats, will be converted by client lib
            "num_leaves_to_search_param": num_leaves_to_search_param,
            "k": k
        }
        param_types = {
            "query_vector": spanner_v1.param_types.Array(spanner_v1.param_types.FLOAT32),
            "num_leaves_to_search_param": spanner_v1.param_types.INT64,
            "k": spanner_v1.param_types.INT64
        }

        print(f"Executing APPROX_COSINE_DISTANCE search: {sql_query.strip()}")
        print(f"Parameters: num_leaves_to_search={num_leaves_to_search_param}, k={k}")

        try:
            with self.database.snapshot() as snapshot:
                results = snapshot.execute_sql(sql_query, params=params, param_types=param_types)
                for row in results:
                    results_list.append(row[0])
        except Exception as e:
            print(f"Error during APPROX_COSINE_DISTANCE search: {e}")
            return [] # Return empty if search fails

        return results_list

    def optimize(self, data_size: Optional[int] = None, **kwargs):
        index_name = f"{self.table_name}_embedding_approx_idx" # Consistent index name
        index_params = self.case_config.index_param()
        num_leaves_val = index_params.get('num_leaves')
        tree_depth_val = index_params.get('tree_depth')

        # Drop the index if it exists
        drop_ddl = f"DROP VECTOR INDEX IF EXISTS {index_name}"
        print(f"Attempting to drop vector index (if exists): {drop_ddl}")
        try:
            drop_op = self.database.update_ddl([drop_ddl])
            drop_op.result(timeout=3600) # Wait for completion, potentially long
            print(f"Vector index {index_name} dropped or did not exist.")
        except Exception as e:
            # Log error during drop, but proceed to create, as "IF EXISTS" should handle non-existence for some errors.
            # However, other errors (permissions, DDL structure for specific Spanner versions) could occur.
            print(f"Warning: Error encountered during DROP VECTOR INDEX IF EXISTS: {e}. Attempting creation.")

        # Create the vector index
        # Note: DDL uses `distance_type`, `tree_depth`, `num_leaves` as per the latest request.
        create_ddl = f"""CREATE VECTOR INDEX {index_name}
                       ON {self.table_name}(embedding)
                       OPTIONS (distance_type = 'COSINE', tree_depth = {tree_depth_val}, num_leaves = {num_leaves_val})"""

        print(f"Attempting to create vector index: {create_ddl}")
        try:
            create_op = self.database.update_ddl([create_ddl])
            create_op.result(timeout=3600) # Wait for completion, potentially very long operation
            print(f"Vector index {index_name} creation process completed.")
        except Exception as e_create: # Changed variable name to avoid conflict
            print(f"Error creating vector index {index_name}: {e_create}")
            raise # Re-raise the exception as index creation is critical

    def need_normalize_cosine(self) -> bool:
        # Spanner's COSINE_DISTANCE function works correctly with unnormalized vectors.
        # It calculates 1 - (cosine_similarity).
        # So, no explicit normalization is required by the client *for the distance function itself*.
        return False

    # Helper or additional methods can be added here if necessary
    def __del__(self):
        # Proper cleanup of resources if any were not handled by context managers
        # The Spanner client itself doesn't have an explicit close() method in the same way
        # some other DB clients do. It manages connections internally.
        # If we had created specific sessions or transactions that needed explicit closing
        # outside of a 'with' block, this would be the place.
        # For now, relying on Python's garbage collection for the client object.
        pass

```
