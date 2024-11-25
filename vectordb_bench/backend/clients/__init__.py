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
    Pinecone = "Pinecone"
    ElasticCloud = "ElasticCloud"
    QdrantCloud = "QdrantCloud"
    WeaviateCloud = "WeaviateCloud"
    PgVector = "PgVector"
    PgVectoRS = "PgVectoRS"
    PgVectorScale = "PgVectorScale"
    PgDiskANN = "PgDiskANN"
    AlloyDB = "AlloyDB"
    Redis = "Redis"
    MemoryDB = "MemoryDB"
    Chroma = "Chroma"
    AWSOpenSearch = "OpenSearch"
    Test = "test"


    @property
    def init_cls(self) -> Type[VectorDB]:
        """Import while in use"""
        if self == DB.Milvus:
            from .milvus.milvus import Milvus
            return Milvus

        if self == DB.ZillizCloud:
            from .zilliz_cloud.zilliz_cloud import ZillizCloud
            return ZillizCloud

        if self == DB.Pinecone:
            from .pinecone.pinecone import Pinecone
            return Pinecone

        if self == DB.ElasticCloud:
            from .elastic_cloud.elastic_cloud import ElasticCloud
            return ElasticCloud

        if self == DB.QdrantCloud:
            from .qdrant_cloud.qdrant_cloud import QdrantCloud
            return QdrantCloud

        if self == DB.WeaviateCloud:
            from .weaviate_cloud.weaviate_cloud import WeaviateCloud
            return WeaviateCloud

        if self == DB.PgVector:
            from .pgvector.pgvector import PgVector
            return PgVector

        if self == DB.PgVectoRS:
            from .pgvecto_rs.pgvecto_rs import PgVectoRS
            return PgVectoRS
        
        if self == DB.PgVectorScale:
            from .pgvectorscale.pgvectorscale import PgVectorScale
            return PgVectorScale

        if self == DB.PgDiskANN:
            from .pgdiskann.pgdiskann import PgDiskANN
            return PgDiskANN

        if self == DB.Redis:
            from .redis.redis import Redis
            return Redis
        
        if self == DB.MemoryDB:
            from .memorydb.memorydb import MemoryDB
            return MemoryDB

        if self == DB.Chroma:
            from .chroma.chroma import ChromaClient
            return ChromaClient

        if self == DB.AWSOpenSearch:
            from .aws_opensearch.aws_opensearch import AWSOpenSearch
            return AWSOpenSearch
        
        if self == DB.AlloyDB:
            from .alloydb.alloydb import AlloyDB
            return AlloyDB

    @property
    def config_cls(self) -> Type[DBConfig]:
        """Import while in use"""
        if self == DB.Milvus:
            from .milvus.config import MilvusConfig
            return MilvusConfig

        if self == DB.ZillizCloud:
            from .zilliz_cloud.config import ZillizCloudConfig
            return ZillizCloudConfig

        if self == DB.Pinecone:
            from .pinecone.config import PineconeConfig
            return PineconeConfig

        if self == DB.ElasticCloud:
            from .elastic_cloud.config import ElasticCloudConfig
            return ElasticCloudConfig

        if self == DB.QdrantCloud:
            from .qdrant_cloud.config import QdrantConfig
            return QdrantConfig

        if self == DB.WeaviateCloud:
            from .weaviate_cloud.config import WeaviateConfig
            return WeaviateConfig

        if self == DB.PgVector:
            from .pgvector.config import PgVectorConfig
            return PgVectorConfig

        if self == DB.PgVectoRS:
            from .pgvecto_rs.config import PgVectoRSConfig
            return PgVectoRSConfig

        if self == DB.PgVectorScale:
            from .pgvectorscale.config import PgVectorScaleConfig
            return PgVectorScaleConfig

        if self == DB.PgDiskANN:
            from .pgdiskann.config import PgDiskANNConfig
            return PgDiskANNConfig

        if self == DB.Redis:
            from .redis.config import RedisConfig
            return RedisConfig
        
        if self == DB.MemoryDB:
            from .memorydb.config import MemoryDBConfig
            return MemoryDBConfig

        if self == DB.Chroma:
            from .chroma.config import ChromaConfig
            return ChromaConfig

        if self == DB.AWSOpenSearch:
            from .aws_opensearch.config import AWSOpenSearchConfig
            return AWSOpenSearchConfig
        
        if self == DB.AlloyDB:
            from .alloydb.config import AlloyDBConfig
            return AlloyDBConfig

    def case_config_cls(self, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        if self == DB.Milvus:
            from .milvus.config import _milvus_case_config
            return _milvus_case_config.get(index_type)

        if self == DB.ZillizCloud:
            from .zilliz_cloud.config import AutoIndexConfig
            return AutoIndexConfig

        if self == DB.ElasticCloud:
            from .elastic_cloud.config import ElasticCloudIndexConfig
            return ElasticCloudIndexConfig

        if self == DB.QdrantCloud:
            from .qdrant_cloud.config import QdrantIndexConfig
            return QdrantIndexConfig

        if self == DB.WeaviateCloud:
            from .weaviate_cloud.config import WeaviateIndexConfig
            return WeaviateIndexConfig

        if self == DB.PgVector:
            from .pgvector.config import _pgvector_case_config
            return _pgvector_case_config.get(index_type)

        if self == DB.PgVectoRS:
            from .pgvecto_rs.config import _pgvecto_rs_case_config
            return _pgvecto_rs_case_config.get(index_type)

        if self == DB.AWSOpenSearch:
            from .aws_opensearch.config import AWSOpenSearchIndexConfig
            return AWSOpenSearchIndexConfig

        if self == DB.PgVectorScale:
            from .pgvectorscale.config import _pgvectorscale_case_config
            return _pgvectorscale_case_config.get(index_type)

        if self == DB.PgDiskANN:
            from .pgdiskann.config import _pgdiskann_case_config
            return _pgdiskann_case_config.get(index_type)
        
        if self == DB.AlloyDB:
            from .alloydb.config import _alloydb_case_config
            return _alloydb_case_config.get(index_type)

        # DB.Pinecone, DB.Chroma, DB.Redis
        return EmptyDBCaseConfig


__all__ = [
    "DB", "VectorDB", "DBConfig", "DBCaseConfig", "IndexType", "MetricType", "EmptyDBCaseConfig",
]
