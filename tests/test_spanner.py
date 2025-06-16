import unittest
from unittest.mock import MagicMock, patch, ANY, call
import os

# Set emulator host for tests if not set, to avoid credential errors during object init
if "SPANNER_EMULATOR_HOST" not in os.environ:
    os.environ["SPANNER_EMULATOR_HOST"] = "localhost:9010"

# Import spanner_v1 for param_types used in assertions
from google.cloud import spanner_v1
from vectordb_bench.backend.clients.spanner.spanner import SpannerVectorDB
from vectordb_bench.backend.clients.spanner.config import SpannerConfig, SpannerCaseConfig
# EmptyDBCaseConfig might not be needed if SpannerCaseConfig is always used.
# from vectordb_bench.backend.clients.api import EmptyDBCaseConfig


class TestSpannerCaseConfig(unittest.TestCase):
    def test_index_param(self):
        config = SpannerCaseConfig(num_leaves=1500, tree_depth=3, num_leaves_to_search=150)
        expected = {"num_leaves": 1500, "tree_depth": 3}
        self.assertEqual(config.index_param(), expected)

    def test_search_param(self):
        config = SpannerCaseConfig(num_leaves=1500, tree_depth=3, num_leaves_to_search=150)
        expected = {"num_leaves_to_search": 150}
        self.assertEqual(config.search_param(), expected)


class TestSpannerVectorDB(unittest.TestCase):
    PROJECT_ID = "test-project"
    INSTANCE_ID = "test-instance"
    DATABASE_ID = "test-database"
    TABLE_NAME = "test_vector_embeddings"
    DIMENSION = 128

    def get_db_config(self):
        return {
            "project_id": self.PROJECT_ID,
            "instance_id": self.INSTANCE_ID,
            "database_id": self.DATABASE_ID,
            "table_name": self.TABLE_NAME,
            # "emulator_host": os.environ.get("SPANNER_EMULATOR_HOST") # Handled by SpannerVectorDB
        }

    def get_case_config(self, num_leaves=2000, num_leaves_to_search=50, tree_depth=2):
        # Using SpannerCaseConfig directly and allowing parameterization for tests
        return SpannerCaseConfig(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            tree_depth=tree_depth
        )

    @patch('vectordb_bench.backend.clients.spanner.spanner.spanner_v1.Client')
    def setUp(self, MockSpannerClient):
        # Mock the Spanner client and its hierarchy
        self.mock_client = MockSpannerClient.return_value
        self.mock_instance = MagicMock()
        self.mock_database = MagicMock()

        self.mock_client.instance.return_value = self.mock_instance
        self.mock_instance.database.return_value = self.mock_database

        # Simulate instance and database exist by default
        self.mock_instance.exists.return_value = True
        self.mock_database.exists.return_value = True

        # Mock DDL operations
        self.mock_ddl_operation = MagicMock()
        self.mock_ddl_operation.result.return_value = None # Simulate DDL completion
        self.mock_database.update_ddl.return_value = self.mock_ddl_operation

        # Mock snapshot and batch
        self.mock_snapshot_context = MagicMock()
        self.mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value = self.mock_snapshot_context
        self.mock_snapshot_context.__enter__.return_value = self.mock_snapshot
        self.mock_snapshot_context.__exit__.return_value = None

        self.mock_batch_context = MagicMock()
        self.mock_batch = MagicMock()
        self.mock_database.batch.return_value = self.mock_batch_context
        self.mock_batch_context.__enter__.return_value = self.mock_batch
        self.mock_batch_context.__exit__.return_value = None

        # Default to table not existing for initial setup in most tests
        self.mock_snapshot.execute_sql.side_effect = [iter([])] # No results for table check

        # Initialize SpannerVectorDB
        # We pass SpannerConfig directly now, not a dict
        self.db = SpannerVectorDB(
            dim=self.DIMENSION,
            db_config=self.get_db_config(),
            db_case_config=self.get_case_config(),
            collection_name=self.TABLE_NAME, # collection_name is used as table_name by SpannerVectorDB
            drop_old=False
        )
        # Reset side effect for subsequent execute_sql calls in tests
        self.mock_snapshot.execute_sql.side_effect = None


    @patch('vectordb_bench.backend.clients.spanner.spanner.spanner_v1.Client')
    def test_init_emulator(self, MockSpannerClient):
        os.environ["SPANNER_EMULATOR_HOST"] = "localhost:9010"
        db_config_dict = self.get_db_config()
        db_config_dict["emulator_host"] = "localhost:9010"

        db = SpannerVectorDB(
            dim=self.DIMENSION,
            db_config=db_config_dict,
            db_case_config=self.get_case_config(),
            collection_name=self.TABLE_NAME,
            drop_old=False
        )
        MockSpannerClient.assert_called_with(project=self.PROJECT_ID, client_options={'api_endpoint': 'localhost:9010'})
        del os.environ["SPANNER_EMULATOR_HOST"] # clean up

    @patch('vectordb_bench.backend.clients.spanner.spanner.spanner_v1.Client')
    def test_init_no_emulator(self, MockSpannerClient):
        if "SPANNER_EMULATOR_HOST" in os.environ: # Ensure it's not set for this test
            del os.environ["SPANNER_EMULATOR_HOST"]

        db_config_dict = self.get_db_config()
        # db_config_dict["emulator_host"] = None # Ensure it's None or not present

        db = SpannerVectorDB(
            dim=self.DIMENSION,
            db_config=db_config_dict,
            db_case_config=self.get_case_config(),
            collection_name=self.TABLE_NAME,
            drop_old=False
        )
        MockSpannerClient.assert_called_with(project=self.PROJECT_ID)
        # Restore for other tests if it was globally set
        if "SPANNER_EMULATOR_HOST" not in os.environ and self.db.emulator_host:
             os.environ["SPANNER_EMULATOR_HOST"] = self.db.emulator_host


    def test_create_table_ddl(self):
        # This test relies on self.db initialized in setUp,
        # where table_exists mock returns False initially.
        # _create_table_if_not_exists is called during SpannerVectorDB.__init__
        self.mock_database.update_ddl.assert_called_once()

        ddl_statements_list = self.mock_database.update_ddl.call_args[0][0]
        self.assertEqual(len(ddl_statements_list), 1) # Only CREATE TABLE DDL

        table_ddl = ddl_statements_list[0]
        self.assertIn(f"CREATE TABLE {self.TABLE_NAME}", table_ddl)
        self.assertIn(f"embedding ARRAY<FLOAT32>(vector_length={self.DIMENSION}) NOT NULL", table_ddl)
        self.assertNotIn("NOT_NULLABLE_embedding", table_ddl)
        self.assertNotIn("CONSTRAINT chk_embedding_dimension", table_ddl)
        self.assertNotIn("CREATE VECTOR INDEX", table_ddl) # Index creation moved to optimize

    def test_create_table_if_not_exists_already_exists(self):
        # Simulate table already exists
        self.mock_snapshot.execute_sql.side_effect = [iter([("sometable",)])] # Table exists
        self.mock_database.reset_mock() # Reset from setUp's potential call

        # Re-initialize SpannerVectorDB for this specific scenario
        db = SpannerVectorDB(
            dim=self.DIMENSION,
            db_config=self.get_db_config(),
            db_case_config=self.get_case_config(), # Use default case config
            collection_name=self.TABLE_NAME,
            drop_old=False
        )
        self.mock_database.update_ddl.assert_not_called() # Should not try to create

    def test_init_drop_old_table_exists(self):
        # Mock sequence: 1. table_exists (for drop) = True, 2. table_exists (for create) = False
        self.mock_snapshot.execute_sql.side_effect = [
            iter([("sometable",)]),  # Table exists for _drop_table
            iter([])                 # Table does not exist for _create_table_if_not_exists
        ]
        self.mock_database.reset_mock()
        self.mock_database.update_ddl.return_value = self.mock_ddl_operation

        db = SpannerVectorDB(
            dim=self.DIMENSION,
            db_config=self.get_db_config(),
            db_case_config=self.get_case_config(), # Use default case config
            collection_name=self.TABLE_NAME,
            drop_old=True
        )

        self.assertEqual(self.mock_database.update_ddl.call_count, 2)

        # Call 1: DROP TABLE
        drop_ddl_list = self.mock_database.update_ddl.call_args_list[0][0][0]
        self.assertEqual(len(drop_ddl_list), 1)
        self.assertIn(f"DROP TABLE {self.TABLE_NAME}", drop_ddl_list[0])

        # Call 2: CREATE TABLE (only, no index here)
        create_ddl_list = self.mock_database.update_ddl.call_args_list[1][0][0]
        self.assertEqual(len(create_ddl_list), 1)
        self.assertIn(f"CREATE TABLE {self.TABLE_NAME}", create_ddl_list[0])
        self.assertIn(f"embedding ARRAY<FLOAT32>(vector_length={self.DIMENSION}) NOT NULL", create_ddl_list[0])
        self.assertNotIn("CREATE VECTOR INDEX", create_ddl_list[0])


    def test_optimize_ddl(self):
        custom_num_leaves = 1234
        custom_tree_depth = 4
        case_config = self.get_case_config(num_leaves=custom_num_leaves, tree_depth=custom_tree_depth)
        self.db.case_config = case_config # Override for this test

        # Reset mock from setUp because __init__ calls _create_table_if_not_exists
        self.mock_database.reset_mock()
        self.mock_database.update_ddl.return_value = self.mock_ddl_operation # re-assign

        with self.db.init():
            self.db.optimize()

        self.assertEqual(self.mock_database.update_ddl.call_count, 2)

        # Call 1: DROP VECTOR INDEX
        drop_index_ddl_list = self.mock_database.update_ddl.call_args_list[0][0][0]
        self.assertEqual(len(drop_index_ddl_list), 1)
        expected_index_name = f"{self.TABLE_NAME}_embedding_approx_idx"
        self.assertIn(f"DROP VECTOR INDEX IF EXISTS {expected_index_name}", drop_index_ddl_list[0])

        # Call 2: CREATE VECTOR INDEX
        create_index_ddl_list = self.mock_database.update_ddl.call_args_list[1][0][0]
        self.assertEqual(len(create_index_ddl_list), 1)
        create_index_ddl = create_index_ddl_list[0]
        self.assertIn(f"CREATE VECTOR INDEX {expected_index_name}", create_index_ddl)
        self.assertIn(f"ON {self.TABLE_NAME}(embedding)", create_index_ddl)
        self.assertIn(f"OPTIONS (distance_type = 'COSINE', tree_depth = {custom_tree_depth}, num_leaves = {custom_num_leaves})", create_index_ddl)


    def test_insert_embeddings(self):
        embeddings = [[float(i) for i in range(self.DIMENSION)] for _ in range(10)]
        metadata = [i for i in range(10)]

        # Reset batch mock for specific call verification
        self.mock_batch.reset_mock()

        with self.db.init(): # Context manager for SpannerVectorDB
            count, error = self.db.insert_embeddings(embeddings, metadata)

        self.assertEqual(count, 10)
        self.assertIsNone(error)
        self.mock_database.batch.assert_called_once()
        self.mock_batch.insert.assert_called_once_with(
            table=self.TABLE_NAME,
            columns=['id', 'embedding'],
            values=[(metadata[i], embeddings[i]) for i in range(10)]
        )

    def test_insert_embeddings_empty(self):
        with self.db.init():
             count, error = self.db.insert_embeddings([], [])
        self.assertEqual(count, 0)
        self.assertIsNone(error)
        self.mock_batch.insert.assert_not_called()

    def test_insert_embeddings_error(self):
        self.mock_batch.insert.side_effect = Exception("Spanner Error")
        embeddings = [[1.0] * self.DIMENSION]
        metadata = [1]

        with self.db.init():
            count, error = self.db.insert_embeddings(embeddings, metadata)

        self.assertEqual(count, 0)
        self.assertIsNotNone(error)
        self.assertIsInstance(error, Exception)

    def test_search_embedding_approx_cosine_distance(self):
        query_vector = [0.5] * self.DIMENSION
        k_val = 7
        custom_leaves_to_search = 65

        # Override case_config for this specific test instance if needed, or use setUp's default
        self.db.case_config = self.get_case_config(num_leaves_to_search=custom_leaves_to_search)

        expected_ids = [101, 102, 103]
        mock_search_results = [(id_val,) for id_val in expected_ids]
        self.mock_snapshot.execute_sql.return_value = iter(mock_search_results)

        with self.db.init():
            results = self.db.search_embedding(query_vector, k=k_val)

        self.assertEqual(results, expected_ids)

        expected_sql = f"""
        SELECT id
        FROM {self.TABLE_NAME}
        ORDER BY APPROX_COSINE_DISTANCE(embedding, @query_vector, options => JSON_OBJECT('num_leaves_to_search', @num_leaves_to_search_param))
        LIMIT @k
        """
        # Strip whitespace for robust comparison
        # self.mock_snapshot.execute_sql.assert_called_once()
        args, kwargs = self.mock_snapshot.execute_sql.call_args
        self.assertEqual(args[0].strip(), expected_sql.strip())

        expected_params = {
            "query_vector": query_vector,
            "num_leaves_to_search_param": custom_leaves_to_search,
            "k": k_val
        }
        self.assertEqual(kwargs['params'], expected_params)

        expected_param_types = {
            "query_vector": spanner_v1.param_types.Array(spanner_v1.param_types.FLOAT32),
            "num_leaves_to_search_param": spanner_v1.param_types.INT64,
            "k": spanner_v1.param_types.INT64
        }
        self.assertEqual(kwargs['param_types'], expected_param_types)

    def test_search_embedding_approx_error(self):
        self.mock_snapshot.execute_sql.side_effect = Exception("Spanner SQL Error")
        query_vector = [0.1] * self.DIMENSION

        with self.db.init():
            results = self.db.search_embedding(query_vector, k=10)

        self.assertEqual(results, [])
        self.mock_snapshot.execute_sql.assert_called_once() # Ensure it was called

    def test_need_normalize_cosine(self):
        self.assertFalse(self.db.need_normalize_cosine())

    def test_optimize(self):
        # Optimize is a no-op, just ensure it runs without error
        try:
            with self.db.init():
                self.db.optimize(data_size=1000)
        except Exception as e:
            self.fail(f"optimize() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
```

A quick note on the `setUp` and initialization:
- I'm patching `spanner_v1.Client` at the class or method level for tests that need to control its instantiation.
- The `SpannerVectorDB` constructor expects the `db_config` to be a dictionary, which it then uses to create `SpannerConfig`.
- `collection_name` passed to `SpannerVectorDB` is used as the table name.
- `SpannerCaseConfig` is passed directly.

There was a slight misunderstanding in my previous plan: `SpannerVectorDB`'s constructor takes `db_config` as a `dict`, not a `SpannerConfig` object directly. The `spanner.py` code confirms this: `self.db_config = SpannerConfig(**db_config)`. I've adjusted the test setup accordingly. Also, `spanner_v1` needs to be imported for `param_types`. I'll add that to the `spanner.py` file.

One more thing, `spanner_v1` is used in `spanner.py` but not imported. I need to fix that in `spanner.py`.
