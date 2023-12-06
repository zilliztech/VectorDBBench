import logging
from contextlib import contextmanager
from typing import Any, Type
from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import RedisConfig
import redis
from redis.commands.search.field import TagField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np


log = logging.getLogger(__name__)
INDEX_NAME = "index"                              # Vector Index Name

class Redis(VectorDB):
    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DBCaseConfig,
            drop_old: bool = False,

            **kwargs
        ):

        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = INDEX_NAME

        # Create a redis connection, if db has password configured, add it to the connection here and in init():
        password=self.db_config["password"]
        conn = redis.Redis(host=self.db_config["host"], port=self.db_config["port"], password=password, db=0)
        

        if drop_old:
            try:
                conn.ft(INDEX_NAME).info()
                conn.ft(INDEX_NAME).dropindex()
            except redis.exceptions.ResponseError:
                drop_old = False
                log.info(f"Redis client drop_old collection: {self.collection_name}")
        
        self.make_index(dim, conn)
        conn.close()
        conn = None

    def make_index(self, vector_dimensions: int, conn: redis.Redis):
        try:
            # check to see if index exists
            conn.ft(INDEX_NAME).info()
        except: 
            schema = (
                TagField("id"),                   
                NumericField("metadata"),              
                VectorField("vector",                  # Vector Field Name
                    "HNSW", {                          # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions,      # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                    }
                ),
            )

            definition = IndexDefinition(index_type=IndexType.HASH)

            rs = conn.ft(INDEX_NAME)
            rs.create_index(schema, definition=definition)

    @contextmanager
    def init(self) -> None:
        """ create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn = redis.Redis(host=self.db_config["host"], port=self.db_config["port"], password=self.db_config["password"], db=0)
        yield
        self.conn.close()
        self.conn = None


    def ready_to_search(self) -> bool:
        """Check if the database is ready to search."""
        pass
    

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        pass


    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        """Insert embeddings into the database.
        Should call self.init() first.
        """

        batch_size = 1000 # Adjust this as needed, but don't make too big
        try:
            with self.conn.pipeline(transaction=False) as pipe:
                for i, embedding in enumerate(embeddings):
                    embedding = np.array(embedding).astype(np.float32)
                    pipe.hset(metadata[i], mapping = {
                        "id": str(metadata[i]),
                        "metadata": metadata[i], 
                        "vector": embedding.tobytes(),
                    })
                    # Execute the pipe so we don't keep too much in memory at once
                    if i % batch_size == 0:
                        res = pipe.execute()

                res = pipe.execute()
                result_len = i + 1
        except Exception as e:
            return 0, e
        
        return result_len, None
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> (list[int]):
        assert self.conn is not None
        
        query_vector = np.array(query).astype(np.float32).tobytes()
        query_obj = Query(f"*=>[KNN {k} @vector $vec as score]").sort_by("score").return_fields("id", "score").paging(0, k).dialect(2)
        query_params = {"vec": query_vector}
        
        if filters:
            # benchmark test filters of format: {'metadata': '>=10000', 'id': 10000}
            # gets exact match for id, and range for metadata if they exist in filters
            id_value = filters.get("id")
            metadata_value = filters.get("metadata")
            if id_value and metadata_value:
                query_obj = Query(f"(@metadata:[{metadata_value} +inf] @id:{ {id_value} })=>[KNN {k} @vector $vec as score]").sort_by("score").return_fields("id", "score").paging(0, k).dialect(2)
            elif id_value:
                #gets exact match for id
                query_obj = Query(f"@id:{ {id_value} }=>[KNN {k} @vector $vec as score]").sort_by("score").return_fields("id", "score").paging(0, k).dialect(2)
            else: #metadata only case, greater than or equal to metadata value
                query_obj = Query(f"@metadata:[{metadata_value} +inf]=>[KNN {k} @vector $vec as score]").sort_by("score").return_fields("id", "score").paging(0, k).dialect(2) 
        res = self.conn.ft(INDEX_NAME).search(query_obj, query_params)
        # doc in res of format {'id': '9831', 'payload': None, 'score': '1.19209289551e-07'}
        return [int(doc["id"]) for doc in res.docs]

    
        
