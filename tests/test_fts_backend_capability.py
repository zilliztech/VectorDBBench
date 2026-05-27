import pytest

from vectordb_bench.backend.clients.api import VectorDB
from vectordb_bench.backend.workload import WorkloadKind


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
