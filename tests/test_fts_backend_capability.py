import pytest

from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, VectorDB
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferFtsConfig
from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig
from vectordb_bench.backend.workload import WorkloadKind
from vectordb_bench.models import CaseConfig, TaskConfig


def test_workload_kind_values_are_stable():
    assert WorkloadKind.VECTOR.value == "vector"
    assert WorkloadKind.FULL_TEXT_BM25.value == "full_text_bm25"
    assert WorkloadKind.HYBRID_DENSE_BM25.value == "hybrid_dense_bm25"


def test_vectordb_defaults_to_no_full_text_support():
    assert VectorDB.supports_full_text_search() is False


class FakeVectorDB(VectorDB):
    def __init__(self, *args, **kwargs):
        self.name = "FakeVectorDB"

    def init(self):
        raise NotImplementedError

    def insert_embeddings(self, embeddings, metadata, labels_data=None, **kwargs):
        raise NotImplementedError

    def search_embedding(self, query, k=100):
        raise NotImplementedError

    def optimize(self, data_size=None):
        raise NotImplementedError


def test_default_document_methods_fail_fast():
    db = FakeVectorDB(dim=0, db_config={}, db_case_config=None, collection_name="test")

    with pytest.raises(NotImplementedError, match="does not support full-text document insert"):
        db.insert_documents(texts=["hello"], doc_ids=["doc-1"])

    with pytest.raises(NotImplementedError, match="does not support full-text document search"):
        db.search_documents(query="hello", k=10)


def test_supported_fts_dbs_declare_capability():
    assert DB.Milvus.init_cls.supports_full_text_search()
    assert DB.ElasticCloud.init_cls.supports_full_text_search()
    assert DB.Vespa.init_cls.supports_full_text_search()
    assert DB.TurboPuffer.init_cls.supports_full_text_search()


def test_supported_fts_dbs_map_fts_case_config_by_index_type():
    assert DB.Milvus.case_config_cls(IndexType.FTS_AUTOINDEX) is MilvusFtsConfig
    assert DB.ElasticCloud.case_config_cls(IndexType.FTS_AUTOINDEX) is ElasticCloudFtsConfig
    assert DB.Vespa.case_config_cls(IndexType.FTS_AUTOINDEX) is VespaFtsConfig
    assert DB.TurboPuffer.case_config_cls(IndexType.FTS_AUTOINDEX) is TurboPufferFtsConfig


def test_unsupported_fts_db_fails_at_assembly():
    task = TaskConfig(
        db=DB.QdrantLocal,
        db_config=DB.QdrantLocal.config_cls(url="http://localhost:6333"),
        db_case_config=DB.QdrantLocal.case_config_cls()(m=16, ef_construct=100),
        case_config=CaseConfig(case_id=CaseType.FTSmsmarcoPerformance),
    )

    with pytest.raises(ValueError, match="does not support full-text search"):
        Assembler.assemble("run", task, source=None)
