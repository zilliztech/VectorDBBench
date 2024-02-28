import logging
from vectordb_bench.models import (
    DB,
)
from vectordb_bench.backend.clients.chroma.config import ChromaConfig
import numpy as np
import chromadb


log = logging.getLogger(__name__)

""" Tests for Chroma, assumes Chroma is running on localhost:8000, 
    Chroma docs: https://docs.trychroma.com/usage-guide
    To configure Chroma to run in a docker container client/server
    
    To get running: clone chroma repo and run docker-compose up in chroma directory:
    1. git clone chroma repo https://github.com/chroma-core/chroma
    2. cd chroma, docker-compose up -d --build  # start chroma server
    3. default port is 8000, default host is localhost"""



dict = {} #Assumes chroma is acception connections on localhost:8000
dict['name'] = "chroma"
dict['host'] = "localhost"
dict['port'] = 8000
dict['password'] = "chroma"



class TestChroma:
    def test_insert_and_search(self):
        assert DB.Chroma.value == "Chroma"

        dbcls = DB.Chroma.init_cls
        dbConfig = DB.Chroma.config_cls
        

        dim = 16
        chrma = dbcls(
            dim=dim,
            db_config=dict,
            db_case_config=None,
            indice="example",
            drop_old=True,
        )

        count = 10_000
        filter_value = 0.9
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]


        # insert
        with chrma.init():
            #chrma.client.delete_collection("example2")
            assert (chrma.client.heartbeat() is not None), "chroma client is not connected"
            res = chrma.insert_embeddings(embeddings=embeddings, metadata=range(count))
            # bulk_insert return
            assert (
                res[0] == count
            ), f"the return count of bulk insert ({res}) is not equal to count ({count})"

            # count entries in chroma database
            countRes = chrma.collection.count()
            
            assert (
                countRes == count
            ), f"the return count of redis client ({countRes}) is not equal to count ({count})"

        # search
        with chrma.init():
            test_id = np.random.randint(count)
            #log.info(f"test_id: {test_id}")
            q = embeddings[test_id]

            res = chrma.search_embedding(query=q, k=100)
            print(res)
            assert (
                res[0] == int(test_id)
            ), f"the most nearest neighbor ({res[0]}) id is not test_id ({int(test_id)}"
            

        # search with filters, assumes filter format {id: int, metadata: >=int}
        with chrma.init():
            filter_value = int(count * filter_value)
            test_id = np.random.randint(filter_value, count)
            q = embeddings[test_id]


            res = chrma.search_embedding(
                query=q, k=100, filters={"id": filter_value}
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
            assert isFilter, f"Filter not working, id_list: {id_list}"

            