import tempfile

import numpy as np

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.infino.config import InfinoFTSConfig
from vectordb_bench.backend.filter import IntFilter, LabelFilter


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

    def test_numge_filter(self):
        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        dim = 16
        count = 1_000
        threshold = 500
        rng = np.random.default_rng(1)
        embeddings = rng.random((count, dim)).tolist()

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=dim,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=case_config_cls(metric_type=MetricType.L2, n_cent=8, nprobe=8),
                collection_name="test_numge",
                drop_old=True,
            )
            with client.init():
                client.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))

            with client.init():
                client.prepare_filter(IntFilter(filter_rate=0.5, int_field="id", int_value=threshold))
                query_id = 700
                res = client.search_embedding(query=embeddings[query_id], k=10)
                assert res[0] == query_id
                assert all(r >= threshold for r in res), f"NumGE leaked ids < {threshold}: {res}"

    def test_strequal_filter(self):
        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls
        case_config_cls = DB.Infino.case_config_cls()

        dim = 16
        count = 1_000
        rng = np.random.default_rng(2)
        embeddings = rng.random((count, dim)).tolist()
        label_filter = LabelFilter(label_percentage=0.5)
        target = label_filter.label_value
        # Even ids carry the target label; odd ids get a different one.
        labels = [target if i % 2 == 0 else "label_other" for i in range(count)]

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=dim,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=case_config_cls(metric_type=MetricType.L2, n_cent=8, nprobe=8),
                collection_name="test_strequal",
                drop_old=True,
                with_scalar_labels=True,
            )
            with client.init():
                client.insert_embeddings(
                    embeddings=embeddings,
                    metadata=list(range(count)),
                    labels_data=labels,
                )

            with client.init():
                client.prepare_filter(label_filter)
                query_id = 200  # even -> carries target label
                res = client.search_embedding(query=embeddings[query_id], k=10)
                assert res[0] == query_id
                assert all(r % 2 == 0 for r in res), f"StrEqual leaked non-target rows: {res}"

    def test_fts_bm25(self):
        assert DB.Infino.init_cls.supports_full_text_search() is True

        dbcls = DB.Infino.init_cls
        config_cls = DB.Infino.config_cls

        docs = ["alpha beta", "gamma delta", "beta gamma", "unique zebra sentence"]
        doc_ids = [str(i) for i in range(len(docs))]

        with tempfile.TemporaryDirectory() as data_path:
            client = dbcls(
                dim=0,
                db_config=config_cls(data_path=data_path).to_dict(),
                db_case_config=InfinoFTSConfig(),
                collection_name="test_fts",
                drop_old=True,
            )
            with client.init():
                inserted, err = client.insert_documents(texts=docs, doc_ids=doc_ids)
                assert err is None
                assert inserted == len(docs)

            with client.init():
                res = client.search_documents(query="zebra", k=10)
                assert res == ["3"], f"expected doc 3 for 'zebra', got {res}"
                # returned ids are strings matching the FTS ground-truth dtype
                assert all(isinstance(r, str) for r in res)
