from enum import Enum

from .api import (
    DBCaseConfig,
    DBConfig,
    EmptyDBCaseConfig,
    IndexType,
    MetricType,
    VectorDB,
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
    AliyunElasticsearch = "AliyunElasticsearch"
    MariaDB = "MariaDB"
    Test = "test"
    AliyunOpenSearch = "AliyunOpenSearch"
    MongoDB = "MongoDB"
    TiDB = "TiDB"
    Clickhouse = "Clickhouse"
    Vespa = "Vespa"
    LanceDB = "LanceDB"

    @property
    def init_cls(self) -> type[VectorDB]:  # noqa: PLR0911, PLR0912, C901, PLR0915
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

        if self == DB.Clickhouse:
            from .clickhouse.clickhouse import Clickhouse

            return Clickhouse

        if self == DB.AlloyDB:
            from .alloydb.alloydb import AlloyDB

            return AlloyDB

        if self == DB.AliyunElasticsearch:
            from .aliyun_elasticsearch.aliyun_elasticsearch import AliyunElasticsearch

            return AliyunElasticsearch

        if self == DB.AliyunOpenSearch:
            from .aliyun_opensearch.aliyun_opensearch import AliyunOpenSearch

            return AliyunOpenSearch

        if self == DB.MongoDB:
            from .mongodb.mongodb import MongoDB

            return MongoDB

        if self == DB.MariaDB:
            from .mariadb.mariadb import MariaDB

            return MariaDB

        if self == DB.TiDB:
            from .tidb.tidb import TiDB

            return TiDB

        if self == DB.Test:
            from .test.test import Test

            return Test

        if self == DB.Vespa:
            from .vespa.vespa import Vespa

            return Vespa

        if self == DB.LanceDB:
            from .lancedb.lancedb import LanceDB

            return LanceDB

        msg = f"Unknown DB: {self.name}"
        raise ValueError(msg)

    @property
    def config_cls(self) -> type[DBConfig]:  # noqa: PLR0911, PLR0912, C901, PLR0915
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

        if self == DB.Clickhouse:
            from .clickhouse.config import ClickhouseConfig

            return ClickhouseConfig

        if self == DB.AlloyDB:
            from .alloydb.config import AlloyDBConfig

            return AlloyDBConfig

        if self == DB.AliyunElasticsearch:
            from .aliyun_elasticsearch.config import AliyunElasticsearchConfig

            return AliyunElasticsearchConfig

        if self == DB.AliyunOpenSearch:
            from .aliyun_opensearch.config import AliyunOpenSearchConfig

            return AliyunOpenSearchConfig

        if self == DB.MongoDB:
            from .mongodb.config import MongoDBConfig

            return MongoDBConfig

        if self == DB.MariaDB:
            from .mariadb.config import MariaDBConfig

            return MariaDBConfig

        if self == DB.TiDB:
            from .tidb.config import TiDBConfig

            return TiDBConfig

        if self == DB.Test:
            from .test.config import TestConfig

            return TestConfig

        if self == DB.Vespa:
            from .vespa.config import VespaConfig

            return VespaConfig

        if self == DB.LanceDB:
            from .lancedb.config import LanceDBConfig

            return LanceDBConfig

        msg = f"Unknown DB: {self.name}"
        raise ValueError(msg)

    def case_config_cls(  # noqa: C901, PLR0911, PLR0912
        self,
        index_type: IndexType | None = None,
    ) -> type[DBCaseConfig]:
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

        if self == DB.Clickhouse:
            from .clickhouse.config import ClickhouseHNSWConfig

            return ClickhouseHNSWConfig

        if self == DB.PgVectorScale:
            from .pgvectorscale.config import _pgvectorscale_case_config

            return _pgvectorscale_case_config.get(index_type)

        if self == DB.PgDiskANN:
            from .pgdiskann.config import _pgdiskann_case_config

            return _pgdiskann_case_config.get(index_type)

        if self == DB.AlloyDB:
            from .alloydb.config import _alloydb_case_config

            return _alloydb_case_config.get(index_type)

        if self == DB.AliyunElasticsearch:
            from .elastic_cloud.config import ElasticCloudIndexConfig

            return ElasticCloudIndexConfig

        if self == DB.AliyunOpenSearch:
            from .aliyun_opensearch.config import AliyunOpenSearchIndexConfig

            return AliyunOpenSearchIndexConfig

        if self == DB.MongoDB:
            from .mongodb.config import MongoDBIndexConfig

            return MongoDBIndexConfig

        if self == DB.MariaDB:
            from .mariadb.config import _mariadb_case_config

            return _mariadb_case_config.get(index_type)

        if self == DB.TiDB:
            from .tidb.config import TiDBIndexConfig

            return TiDBIndexConfig

        if self == DB.Vespa:
            from .vespa.config import VespaHNSWConfig

            return VespaHNSWConfig

        if self == DB.LanceDB:
            from .lancedb.config import _lancedb_case_config

            return _lancedb_case_config.get(index_type)

        # DB.Pinecone, DB.Chroma, DB.Redis
        return EmptyDBCaseConfig


__all__ = [
    "DB",
    "DBCaseConfig",
    "DBConfig",
    "EmptyDBCaseConfig",
    "IndexType",
    "MetricType",
    "VectorDB",
]
