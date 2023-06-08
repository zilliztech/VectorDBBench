from enum import Enum
from typing import Type
from .api import (
    VectorDB,
    DBConfig,
    DBCaseConfig,
    EmptyDBCaseConfig,
    IndexType,
    MetricType,
)

from .milvus.milvus import Milvus
from .es.elasticsearch import Elasticsearch
from .pinecone.pinecone import Pinecone
from .weaviate.weaviate import Weaviate
from .qdrant.qdrant import Qdrant
from .zilliz_cloud.zilliz_cloud import ZillizCloud


class DB(Enum):
    """Database types

    Examples:
        >>> DB.Milvus
        <DB.Milvus: 'Milvus'>
        >>> DB.Milvus.value
        "Milvus"
        >>> DB.Milvus.name
        "Milvus"
    """

    Milvus = "Milvus"
    ZillizCloud = "ZillizCloud"
    Weaviate = "Weaviate"
    Elasticsearch = "Elasticsearch"
    Qdrant = "Qdrant"
    Pinecone = "Pinecone"


    @property
    def init_cls(self) -> Type[VectorDB]:
        return db2client.get(self)


db2client = {
    DB.Milvus: Milvus,
    DB.ZillizCloud: ZillizCloud,
    DB.Weaviate: Weaviate,
    DB.Elasticsearch: Elasticsearch,
    DB.Qdrant: Qdrant,
    DB.Pinecone: Pinecone,
}


__all__ = [
    "DB", "VectorDB", "DBConfig", "DBCaseConfig", "IndexType", "MetricType", "EmptyDBCaseConfig",
]
