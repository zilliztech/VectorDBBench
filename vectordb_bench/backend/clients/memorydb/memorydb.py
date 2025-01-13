import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import redis
from redis import Redis
from redis.cluster import RedisCluster
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.query import Query

from ..api import IndexType, VectorDB
from .config import MemoryDBIndexConfig

log = logging.getLogger(__name__)
INDEX_NAME = "index"  # Vector Index Name


class MemoryDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MemoryDBIndexConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = INDEX_NAME
        self.target_nodes = RedisCluster.RANDOM if not self.db_config["cmd"] else None
        self.insert_batch_size = db_case_config.insert_batch_size
        self.dbsize = kwargs.get("num_rows")

        # Create a MemoryDB connection, if db has password configured, add it to the connection here and in init():
        log.info(f"Establishing connection to: {self.db_config}")
        conn = self.get_client(primary=True)
        log.info(f"Connection established: {conn}")
        log.info(conn.execute_command("INFO server"))

        if drop_old:
            try:
                log.info(f"MemoryDB client getting info for: {INDEX_NAME}")
                info = conn.ft(INDEX_NAME).info()
                log.info(f"Index info: {info}")
            except redis.exceptions.ResponseError as e:
                log.warning(e)
                drop_old = False
                log.info(f"MemoryDB client drop_old collection: {self.collection_name}")

            log.info("Executing FLUSHALL")
            conn.flushall()

            # Since the default behaviour of FLUSHALL is asynchronous, wait for db to be empty
            self.wait_until(self.wait_for_empty_db, 3, "", conn)
            if not self.db_config["cmd"]:
                replica_clients = self.get_client(replicas=True)
                for rc, host in replica_clients:
                    self.wait_until(self.wait_for_empty_db, 3, "", rc)
                    log.debug(f"Flushall done in the host: {host}")
                    rc.close()

        self.make_index(dim, conn)
        conn.close()
        conn = None

    def make_index(self, vector_dimensions: int, conn: redis.Redis):
        try:
            # check to see if index exists
            conn.ft(INDEX_NAME).info()
        except Exception as e:
            log.warning(f"Error getting info for index '{INDEX_NAME}': {e}")
            index_param = self.case_config.index_param()
            search_param = self.case_config.search_param()
            vector_parameters = {  # Vector Index Type: FLAT or HNSW
                "TYPE": "FLOAT32",
                "DIM": vector_dimensions,  # Number of Vector Dimensions
                "DISTANCE_METRIC": index_param["metric"],  # Vector Search Distance Metric
            }
            if index_param["m"]:
                vector_parameters["M"] = index_param["m"]
            if index_param["ef_construction"]:
                vector_parameters["EF_CONSTRUCTION"] = index_param["ef_construction"]
            if search_param["ef_runtime"]:
                vector_parameters["EF_RUNTIME"] = search_param["ef_runtime"]

            schema = (
                TagField("id"),
                NumericField("metadata"),
                VectorField(
                    "vector",  # Vector Field Name
                    "HNSW",
                    vector_parameters,
                ),
            )

            definition = IndexDefinition(index_type=IndexType.HASH)
            rs = conn.ft(INDEX_NAME)
            rs.create_index(schema, definition=definition)

    def get_client(self, **kwargs):
        """
        Gets either cluster connection or normal connection based on `cmd` flag.
        CMD stands for Cluster Mode Disabled and is a "mode".
        """
        if not self.db_config["cmd"]:
            # Cluster mode enabled

            client = RedisCluster(
                host=self.db_config["host"],
                port=self.db_config["port"],
                ssl=self.db_config["ssl"],
                password=self.db_config["password"],
                ssl_ca_certs=self.db_config["ssl_ca_certs"],
                ssl_cert_reqs=None,
            )

            # Ping all nodes to create a connection
            client.execute_command("PING", target_nodes=RedisCluster.ALL_NODES)
            replicas = client.get_replicas()

            if len(replicas) > 0:
                # FT.SEARCH is a keyless command, use READONLY for replica connections
                client.execute_command("READONLY", target_nodes=RedisCluster.REPLICAS)

            if kwargs.get("primary", False):
                client = client.get_primaries()[0].redis_connection

            if kwargs.get("replicas", False):
                # Return client and host name for each replica
                return [(c.redis_connection, c.host) for c in replicas]

        else:
            client = Redis(
                host=self.db_config["host"],
                port=self.db_config["port"],
                db=0,
                ssl=self.db_config["ssl"],
                password=self.db_config["password"],
                ssl_ca_certs=self.db_config["ssl_ca_certs"],
                ssl_cert_reqs=None,
            )
            client.execute_command("PING")
        return client

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn = self.get_client()
        search_param = self.case_config.search_param()
        if search_param["ef_runtime"]:
            self.ef_runtime_str = f"EF_RUNTIME {search_param['ef_runtime']}"
        else:
            self.ef_runtime_str = ""
        yield
        self.conn.close()
        self.conn = None

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """

        try:
            with self.conn.pipeline(transaction=False) as pipe:
                for i, embedding in enumerate(embeddings):
                    ndarr_emb = np.array(embedding).astype(np.float32)
                    pipe.hset(
                        metadata[i],
                        mapping={
                            "id": str(metadata[i]),
                            "metadata": metadata[i],
                            "vector": ndarr_emb.tobytes(),
                        },
                    )
                    # Execute the pipe so we don't keep too much in memory at once
                    if (i + 1) % self.insert_batch_size == 0:
                        pipe.execute()

                pipe.execute()
                result_len = i + 1
        except Exception as e:
            return 0, e

        return result_len, None

    def _post_insert(self):
        """Wait for indexing to finish"""
        client = self.get_client(primary=True)
        log.info("Waiting for background indexing to finish")
        args = (self.wait_for_no_activity, 5, "", client)
        self.wait_until(*args)
        if not self.db_config["cmd"]:
            replica_clients = self.get_client(replicas=True)
            for rc, host_name in replica_clients:
                args = (self.wait_for_no_activity, 5, "", rc)
                self.wait_until(*args)
                log.debug(f"Background indexing completed in the host: {host_name}")
                rc.close()

    def wait_until(self, condition: any, interval: int = 5, message: str = "Operation took too long", *args):
        while not condition(*args):
            time.sleep(interval)

    def wait_for_no_activity(self, client: redis.RedisCluster | redis.Redis):
        return client.info("search")["search_background_indexing_status"] == "NO_ACTIVITY"

    def wait_for_empty_db(self, client: redis.RedisCluster | redis.Redis):
        return client.execute_command("DBSIZE") == 0

    def search_embedding(
        self,
        query: list[float],
        k: int = 10,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None

        query_vector = np.array(query).astype(np.float32).tobytes()
        query_obj = Query(f"*=>[KNN {k} @vector $vec]").return_fields("id").paging(0, k)
        query_params = {"vec": query_vector}

        if filters:
            # benchmark test filters of format: {'metadata': '>=10000', 'id': 10000}
            # gets exact match for id, and range for metadata if they exist in filters
            id_value = filters.get("id")
            # Removing '>=' from the id_value: '>=10000'
            metadata_value = filters.get("metadata")[2:]
            if id_value and metadata_value:
                query_obj = (
                    Query(
                        f"(@metadata:[{metadata_value} +inf] @id:{ {id_value} })=>[KNN {k} @vector $vec]",
                    )
                    .return_fields("id")
                    .paging(0, k)
                )
            elif id_value:
                # gets exact match for id
                query_obj = Query(f"@id:{ {id_value} }=>[KNN {k} @vector $vec]").return_fields("id").paging(0, k)
            else:  # metadata only case, greater than or equal to metadata value
                query_obj = (
                    Query(f"@metadata:[{metadata_value} +inf]=>[KNN {k} @vector $vec]").return_fields("id").paging(0, k)
                )
        res = self.conn.ft(INDEX_NAME).search(query_obj, query_params)
        return [int(doc["id"]) for doc in res.docs]
