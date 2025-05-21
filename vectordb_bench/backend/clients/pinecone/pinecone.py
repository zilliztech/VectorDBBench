"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
from contextlib import contextmanager

import pinecone

from ..api import DBCaseConfig, DBConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import PineconeConfig

log = logging.getLogger(__name__)

PINECONE_MAX_NUM_PER_BATCH = 1000
PINECONE_MAX_SIZE_PER_BATCH = 2 * 1024 * 1024  # 2MB


class Pinecone(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.index_name = db_config.get("index_name", "")
        self.api_key = db_config.get("api_key", "")
        self.batch_size = int(
            min(PINECONE_MAX_SIZE_PER_BATCH / (dim * 5), PINECONE_MAX_NUM_PER_BATCH),
        )

        pc = pinecone.Pinecone(api_key=self.api_key)
        index = pc.Index(self.index_name)

        if drop_old:
            index_stats = index.describe_index_stats()
            index_dim = index_stats["dimension"]
            if index_dim != dim:
                msg = f"Pinecone index {self.index_name} dimension mismatch, expected {index_dim} got {dim}"
                raise ValueError(msg)
            for namespace in index_stats["namespaces"]:
                log.info(f"Pinecone index delete namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)

        self._metadata_key = "meta"

    @classmethod
    def config_cls(cls) -> type[DBConfig]:
        return PineconeConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self):
        pc = pinecone.Pinecone(api_key=self.api_key)
        self.index = pc.Index(self.index_name)
        yield

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_datas = []
                for i in range(batch_start_offset, batch_end_offset):
                    insert_data = (
                        str(metadata[i]),
                        embeddings[i],
                        {self._metadata_key: metadata[i]},
                    )
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
    ) -> list[int]:
        pinecone_filters = {} if filters is None else {self._metadata_key: {"$gte": filters["id"]}}
        try:
            res = self.index.query(
                top_k=k,
                vector=query,
                filter=pinecone_filters,
            )["matches"]
        except Exception as e:
            log.warning(f"Error querying index: {e}")
            raise e from e
        return [int(one_res["id"]) for one_res in res]
