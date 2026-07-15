import tempfile

import numpy as np

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType


class TestInfino:
    def test_insert_and_search(self):
        assert DB.Infino.value == "Infino"

        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        dim = 16
        count = 2_000
        rng = np.random.default_rng(0)
        embeddings = rng.random((count, dim)).tolist()

        with tempfile.TemporaryDirectory() as data_path:
            db_config = config_cls(data_path=data_path).to_dict()
            # nprobe == n_cent probes every IVF cell -> exact search for the assertion.
            db_case_config = case_config_cls(metric_type=MetricType.L2, n_cent=8, nprobe=8)

            client = dbcls(
                dim=dim,
                db_config=db_config,
                db_case_config=db_case_config,
                collection_name="test_infino",
                drop_old=True,
            )

            with client.init():
                inserted, err = client.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
                assert err is None
                assert inserted == count

            with client.init():
                test_id = 42
                res = client.search_embedding(query=embeddings[test_id], k=10)
                assert res[0] == test_id, f"nearest neighbor id {res[0]} != query id {test_id}"
