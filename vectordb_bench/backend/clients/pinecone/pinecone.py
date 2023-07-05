"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Type

from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import PineconeConfig


log = logging.getLogger(__name__)

PINECONE_MAX_NUM_PER_BATCH = 1000
PINECONE_MAX_SIZE_PER_BATCH = 2 * 1024 * 1024 # 2MB

class Pinecone(VectorDB):
    def __init__(
        self,
        dim,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.index_name = db_config["index_name"]
        self.api_key = db_config["api_key"]
        self.environment = db_config["environment"]
        self.batch_size = int(min(PINECONE_MAX_SIZE_PER_BATCH / (dim * 5), PINECONE_MAX_NUM_PER_BATCH))
        # Pincone will make connections with server while import
        # so place the import here.
        import pinecone
        pinecone.init(
            api_key=self.api_key, environment=self.environment)
        if drop_old:
            list_indexes = pinecone.list_indexes()
            if self.index_name in list_indexes:
                index = pinecone.Index(self.index_name)
                index_dim = index.describe_index_stats()["dimension"]
                if (index_dim != dim):
                    raise ValueError(
                        f"Pinecone index {self.index_name} dimension mismatch, expected {index_dim} got {dim}")
                log.info(
                    f"Pinecone client delete old index: {self.index_name}")
                index.delete(delete_all=True)
                index.close()
            else:
                raise ValueError(
                    f"Pinecone index {self.index_name} does not exist")

        self._metadata_key = "meta"

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return PineconeConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self) -> None:
        import pinecone
        pinecone.init(
            api_key=self.api_key, environment=self.environment)
        self.index = pinecone.Index(self.index_name)
        yield
        self.index.close()

    def ready_to_load(self):
        pass

    def optimize(self):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_datas = []
                for i in range(batch_start_offset, batch_end_offset):
                    insert_data = (str(metadata[i]), embeddings[i], {
                                self._metadata_key: metadata[i]})
                    insert_datas.append(insert_data)
                self.index.upsert(insert_datas)
                insert_count += batch_end_offset - batch_start_offset
        except Exception as e:
            return (insert_count, e)
        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[tuple[int, float]]:
        if filters is None:
            pinecone_filters = {}
        else:
            pinecone_filters = {self._metadata_key: {"$gte": filters["id"]}}
        try:
            res = self.index.query(
                top_k=k,
                vector=query,
                filter=pinecone_filters,
            )['matches']
        except Exception as e:
            print(f"Error querying index: {e}")
            raise e
        id_res = [int(one_res['id']) for one_res in res]
        return id_res
