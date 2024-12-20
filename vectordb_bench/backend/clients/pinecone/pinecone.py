"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Type
import pinecone
from vectordb_bench.backend.filter import Filter, FilterType
from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import PineconeConfig

import pinecone

from ..api import DBCaseConfig, DBConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import PineconeConfig

log = logging.getLogger(__name__)

PINECONE_MAX_NUM_PER_BATCH = 1000
PINECONE_MAX_SIZE_PER_BATCH = 2 * 1024 * 1024  # 2MB


class Pinecone(VectorDB):
    supported_filter_types: list[FilterType] = [
        FilterType.NonFilter,
        FilterType.Int,
        FilterType.Label,
    ]
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
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

        self.with_scalar_labels = with_scalar_labels
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
        self._scalar_label_field = "label"

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
        labels_data: list[str] = None,
        **kwargs,
    ) -> (int, Exception):
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_datas = []
                for i in range(batch_start_offset, batch_end_offset):
                    metadata_dict = {self._metadata_key: metadata[i]}
                    if self.with_scalar_labels:
                        metadata_dict[self._scalar_label_field] = labels_data[i]
                    insert_data = (
                        str(metadata[i]),
                        embeddings[i],
                        metadata_dict,
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
        timeout: int | None = None,
    ) -> list[int]:
        pinecone_filters = self.expr
        try:
            res = self.index.query(
                top_k=k,
                vector=query,
                filter=pinecone_filters,
            )["matches"]
        except Exception as e:
            print(f"Error querying index: {e}")
            raise e
        id_res = [int(one_res["id"]) for one_res in res]
        return id_res

    def prepare_filter(self, filter: Filter):
        if filter.type == FilterType.NonFilter:
            self.expr = None
        elif filter.type == FilterType.Int:
            self.expr = {self._scalar_id_field: {"$gte": filter.int_value}}
        elif filter.type == FilterType.Label:
            self.expr = {self._scalar_label_field : {"$eq": filter.label_value}}
        else:
            raise ValueError(f"Not support Filter for Pinecone - {filter}")
