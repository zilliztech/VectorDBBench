import pytest
import logging
from vector_db_bench.models import (
    DB,
    IndexType,
    MetricType,
    CaseType,
    ElasticsearchConfig,
    TaskConfig,
    CaseConfig,
    CaseResult,
    TestResult,
    Metric,
)
import numpy as np


log = logging.getLogger(__name__)

cloud_id = ""
password = ""


class TestModels:
    def test_insert_and_search(self):
        assert DB.Elasticsearch.value == "Elasticsearch"
        assert DB.Elasticsearch.config == ElasticsearchConfig

        dbConfig = DB.Elasticsearch.config(cloud_id=cloud_id, password=password)
        dbCaseConfig = DB.Elasticsearch.case_config_cls()(
            metric_type=MetricType.L2, efConstruction=64, M=16, num_candidates=100
        )
        es = DB.Elasticsearch.init_cls(
            db_config=dbConfig.to_dict(), db_case_config=dbCaseConfig, drop_old=True
        )

        dim = 16
        count = 10_000

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        with es.init():
            res = es.insert_embeddings(embeddings=embeddings, metadata=range(count))
            # bulk_insert return
            assert (
                len(res) == count
            ), f"the return count of bulk insert ({len(res)}) is not equal to count ({count})"

            # indice_count return
            es.client.indices.refresh()
            esCountRes = es.client.count(index=es.indice)
            assert (
                esCountRes.raw["count"] == count
            ), f"the return count of es client ({len(res)}) is not equal to count ({count})"

        with es.init():
            test_id = 1235
            q = embeddings[test_id]

            res = es.search_embedding_with_score(query=q, k=100)
            assert (
                res[0][0] == test_id
            ), f"the most nearest neighbor ({res[0][0]}) id is not test_id ({test_id})"
