"""Wrapper around the TencentVDB vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager

from ..api import VectorDB, IndexType
from .config import TencentVDBIndexConfig

import tcvectordb
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams
from tcvectordb.model.enum import FieldType, IndexType, ReadConsistency
from tcvectordb.model.document import Document, SearchParams, Filter

log = logging.getLogger(__name__)


class TencentVDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TencentVDBIndexConfig,
        database_name: str = "test-database",
        collection_name: str = "test-collection",
        int_field_name: str = "int_id",
        drop_old: bool = False,
        name: str = "TencentVDB",
        **kwargs,
    ):
        self.name = name
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.database_name = database_name
        self.collection_name = collection_name
        self.int_field_name = int_field_name
        self.batch_size = 1000

        if drop_old:
            # connect
            log.info("connect to TencentVDB client")
            client = self._connect()

            # drop database
            log.info("drop database")
            for db in client.list_databases():
                if db.database_name == database_name:
                    db.drop_database()

            # create database
            log.info("create database")
            self._create_database(client)

            # create collection
            log.info("create collection")
            self._create_collection(client)

    def _connect(self):
        client = tcvectordb.VectorDBClient(
            url=self.db_config.get("url", ""),
            username=self.db_config.get("username", ""),
            key=self.db_config.get("key", ""),
            read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
            timeout=30,
        )
        # test
        try:
            client.list_databases()
        except Exception as e:
            log.warning(f"{self.name} connect failed.")
            raise e from None
        return client

    def _create_database(self, client):
        try:
            client.create_database(self.database_name)
        except Exception as e:
            log.warning(f"{self.name} create database failed.")
            raise e from None

    def _create_collection(self, client):
        try:
            db = client.database(self.database_name)
            index = Index(
                FilterIndex(
                    name="id",
                    field_type=FieldType.String,
                    index_type=IndexType.PRIMARY_KEY,
                ),
                FilterIndex(
                    name=self.int_field_name,
                    field_type=FieldType.Uint64,
                    index_type=IndexType.FILTER,
                ),
                VectorIndex(
                    name="vector",
                    dimension=self.dim,
                    index_type=self.db_case_config.indexType,
                    metric_type=self.db_case_config.index_param().get(
                        "metric_type", ""
                    ),
                    params=HNSWParams(
                        m=self.db_config.get("m", ""),
                        efconstruction=self.db_config.get("efconstruction", ""),
                    ),
                ),
            )
            db.create_collection(
                name=self.collection_name,
                shard=self.db_config.get("shard", 1),
                replicas=self.db_config.get("replicas", 1),
                description="this is a collection of test embedding",
                index=index,
            )
        except Exception as e:
            log.warning(f"{self.name} create collection failed.")
            raise e from None

    @contextmanager
    def init(self) -> None:
        try:
            log.info("client init")
            client = self._connect()
            log.info("get db")
            db = client.database(self.database_name)
            log.info("get col")
            self.col = db.collection(self.collection_name)
        except:
            log.warning("init failed")

        yield
        self.col = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(
                    batch_start_offset + self.batch_size, len(embeddings)
                )
                documents = [
                    Document(
                        id=f"{metadata[i]}", vector=embeddings[i], int_id=metadata[i]
                    )
                    for i in range(batch_start_offset, batch_end_offset)
                ]
                res = self.col.upsert(
                    documents, build_index=True
                )  # set flase when ivf_flat
                if res:
                    log.warning(res)
        except BaseException as e:
            log.info(f"Failed to insert data: {e}")
            return (0, e)
        except:
            log.warning("what happened?")
        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        filter = (
            Filter(f"{self.int_field_name}{filters.get('metadata')}")
            if filters
            else None
        )
        params = SearchParams(ef=self.db_config.get("ef", 100))
        res = self.col.search(
            vectors=[query],
            filter=filter,
            params=params,
            retrieve_vector=False,
            limit=k,
            output_fields=[self.int_field_name],
        )
        return [item[self.int_field_name] for item in res[0]]

    def optimize(self):
        pass

    def ready_to_load(self):
        pass
