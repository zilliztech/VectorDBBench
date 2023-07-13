import logging
from vectordb_bench.models import (
    DB,
)
from vectordb_bench.backend.clients.redis.config import RedisConfig
import numpy as np


log = logging.getLogger(__name__)

# Tests for Redis, assumes Redis is running on localhost:6379, can be modified by changing the dict below
dict = {}
dict['name'] = "redis"
dict['host'] = "localhost"
dict['port'] = 6379
dict['password'] = "redis"



class TestRedis:
    def test_insert_and_search(self):
        assert DB.Redis.value == "Redis"
        dbcls = DB.Redis.init_cls
        dbConfig = dbcls.config_cls()
        

        dim = 16
        rdb = dbcls(
            dim=dim,
            db_config=dict,
            db_case_config=None,
            indice="test_redis",
            drop_old=True,
        )

        count = 10_000
        filter_value = 0.9
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]


        # insert
        with rdb.init():
            assert (rdb.conn.ping() == True), "redis client is not connected"
            res = rdb.insert_embeddings(embeddings=embeddings, metadata=range(count))
            # bulk_insert return
            assert (
                res[0] == count
            ), f"the return count of bulk insert ({res}) is not equal to count ({count})"

            # count entries in redis database
            countRes = rdb.conn.dbsize()
            
            assert (
                countRes == count
            ), f"the return count of redis client ({countRes}) is not equal to count ({count})"

        # search
        with rdb.init():
            test_id = np.random.randint(count)
            #log.info(f"test_id: {test_id}")
            q = embeddings[test_id]

            res = rdb.search_embedding(query=q, k=100)
            #log.info(f"search_results_id: {res}")
            print(res)
            # res of format [2757, 2944, 8893, 6695, 5571, 608, 455, 3464, 1584, 1807, 8452, 4311...]
            assert (
                res[0] == int(test_id)
            ), f"the most nearest neighbor ({res[0]}) id is not test_id ({str(test_id)}"

        # search with filters
        with rdb.init():
            filter_value = int(count * filter_value)
            test_id = np.random.randint(filter_value, count)
            q = embeddings[test_id]


            res = rdb.search_embedding(
                query=q, k=100, filters={"metadata": filter_value}
            )
            assert (
                res[0] == int(test_id)
            ), f"the most nearest neighbor ({res[0]}) id is not test_id ({test_id})"
            isFilter = True
            id_list = []
            for id in res:
                id_list.append(id)
                if int(id) < filter_value:
                    isFilter = False
                    break
            assert isFilter, f"filters failed, got: ({id}), expected less than ({filter_value})"

            #Test id filter for exact match
            res = rdb.search_embedding(
                query=q, k=100, filters={"id": 9999}
            )
            assert (
                res[0] == 9999
            )

            #Test two filters, id and metadata
            res = rdb.search_embedding(
                query=q, k=100, filters={"metadata": filter_value, "id": 9999}
            )
            assert (
                res[0] == 9999 and len(res) == 1, f"filters failed, got: ({res[0]}), expected ({9999})"
            )