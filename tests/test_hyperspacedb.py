import logging
from vectordb_bench.models import DB
import numpy as np

log = logging.getLogger(__name__)

db_config_dict = {
    "host": "localhost:50051",
    "api_key": "I_LOVE_HYPERSPACEDB"
}

class TestHyperspaceDB:
    def test_insert_and_search(self):
        assert DB.HyperspaceDB.value == "HyperspaceDB"

        dbcls = DB.HyperspaceDB.init_cls

        dim = 16
        db = dbcls(
            dim=dim,
            db_config=db_config_dict,
            db_case_config=None,
            collection_name="example_collection",
            drop_old=True,
        )

        count = 1000
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # insert
        with db.init():
            assert db.client is not None, "Hyperspace client is not connected"
            res = db.insert_embeddings(embeddings=embeddings, metadata=range(count))
            assert res[0] == count, f"Expected {count} inserted, got {res[0]}"

        # optimize
        with db.init():
            db.optimize()

        # search
        with db.init():
            test_id = np.random.randint(count)
            q = embeddings[test_id]

            res = db.search_embedding(query=q, k=10)
            assert len(res) > 0, "No results returned"
            assert int(res[0]) == int(test_id), f"Expected nearest neighbor to be {test_id}, got {res[0]}"
