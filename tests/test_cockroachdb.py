"""
Tests for CockroachDB vector database client.

Assumes CockroachDB is running on localhost:26257.

To start CockroachDB locally:
    cockroach start-single-node --insecure --listen-addr=localhost:26257
"""

import logging

import numpy as np

from vectordb_bench.models import DB

log = logging.getLogger(__name__)


class TestCockroachDB:
    """Test suite for CockroachDB vector operations."""

    def test_insert_and_search(self):
        """Test basic insert and search operations."""
        assert DB.CockroachDB.value == "CockroachDB"

        dbcls = DB.CockroachDB.init_cls
        dbConfig = DB.CockroachDB.config_cls

        # Connection config (matches your local CockroachDB instance)
        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_cockroachdb",
        }
        
        # Note: sslmode=disable is handled in the client's connect_config options

        dim = 128
        count = 1000

        # Initialize CockroachDB client
        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=None,
            collection_name="test_cockroachdb",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # Test insert
        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))

            assert res[0] == count, f"Insert count mismatch: {res[0]} != {count}"
            assert res[1] is None, f"Insert failed with error: {res[1]}"

        # Test search
        with cockroachdb.init():
            test_id = np.random.randint(count)
            q = embeddings[test_id]

            res = cockroachdb.search_embedding(query=q, k=10)

            assert len(res) > 0, "Search returned no results"
            assert res[0] == int(test_id), f"Top result {res[0]} != query id {test_id}"

        log.info("CockroachDB insert and search test passed")

    def test_search_with_filter(self):
        """Test search with filters."""
        assert DB.CockroachDB.value == "CockroachDB"

        dbcls = DB.CockroachDB.init_cls

        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_cockroachdb_filter",
        }

        dim = 128
        count = 1000
        filter_value = 0.9

        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=None,
            collection_name="test_cockroachdb_filter",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # Insert data
        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert res[0] == count, f"Insert count mismatch"

        # Search with filter
        with cockroachdb.init():
            filter_id = int(count * filter_value)
            test_id = np.random.randint(filter_id, count)
            q = embeddings[test_id]

            from vectordb_bench.backend.filter import IntFilter

            filters = IntFilter(int_value=filter_id, filter_rate=0.9)
            cockroachdb.prepare_filter(filters)

            res = cockroachdb.search_embedding(query=q, k=10)

            assert len(res) > 0, "Filtered search returned no results"
            assert res[0] == int(test_id), f"Top result {res[0]} != query id {test_id}"

            # Verify all results are >= filter_value
            for result_id in res:
                assert int(result_id) >= filter_id, f"Result {result_id} < filter threshold {filter_id}"

        log.info("CockroachDB filter test passed")
