import logging
from vectordb_bench.models import (
    DB,
    MetricType,
    ElasticsearchConfig,
)
import numpy as np


log = logging.getLogger(__name__)

cloud_id = ""
password = ""


class TestModels:
    def test_insert_and_search(self):
        assert DB.ElasticCloud.value == "Elasticsearch"
        assert DB.ElasticCloud.config == ElasticsearchConfig

        dbcls = DB.ElasticCloud.init_cls
        dbConfig = DB.ElasticCloud.config_cls(cloud_id=cloud_id, password=password)
        dbCaseConfig = DB.ElasticCloud.case_config_cls()(
            metric_type=MetricType.L2, efConstruction=64, M=16, num_candidates=100
        )

        dim = 16
        es = dbcls(
            dim=dim,
            db_config=dbConfig.to_dict(),
            db_case_config=dbCaseConfig,
            indice="test_es_cloud",
            drop_old=True,
        )

        count = 10_000
        filter_rate = 0.9
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # insert
        with es.init():
            res = es.insert_embeddings(embeddings=embeddings, metadata=range(count))
            # bulk_insert return
            assert (
                res == count
            ), f"the return count of bulk insert ({res}) is not equal to count ({count})"

            # indice_count return
            es.client.indices.refresh()
            esCountRes = es.client.count(index=es.indice)
            countResCount = esCountRes.raw["count"]
            assert (
                countResCount == count
            ), f"the return count of es client ({countResCount}) is not equal to count ({count})"

        # search
        with es.init():
            test_id = np.random.randint(count)
            log.info(f"test_id: {test_id}")
            q = embeddings[test_id]

            res = es.search_embedding(query=q, k=100)
            log.info(f"search_results_id: {res}")
            assert (
                res[0] == test_id
            ), f"the most nearest neighbor ({res[0]}) id is not test_id ({test_id})"

        # search with filters
        with es.init():
            test_id = np.random.randint(count * filter_rate, count)
            log.info(f"test_id: {test_id}")
            q = embeddings[test_id]

            res = es.search_embedding(
                query=q, k=100, filters={"id": count * filter_rate}
            )
            log.info(f"search_results_id: {res}")
            assert (
                res[0] == test_id
            ), f"the most nearest neighbor ({res[0]}) id is not test_id ({test_id})"
            isFilter = True
            for id in res:
                if id < count * filter_rate:
                    isFilter = False
                    break
            assert isFilter, f"filters failed"
