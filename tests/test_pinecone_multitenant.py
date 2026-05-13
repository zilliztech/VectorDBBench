from types import SimpleNamespace


class FakePineconeIndex:
    def __init__(self):
        self.upserts = []
        self.queries = []
        self.deletes = []

    def describe_index_stats(self):
        return {"dimension": 2, "total_vector_count": 3, "namespaces": {"mt_tenant_0000": {}, "mt_tenant_0001": {}}}

    def upsert(self, vectors, namespace=None):
        self.upserts.append((vectors, namespace))
        return SimpleNamespace(_response_info={"raw_headers": {"x-pinecone-request-lsn": "7"}})

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(
            matches=[{"id": "11"}],
            _response_info={"raw_headers": {"x-pinecone-max-indexed-lsn": "7"}},
        )

    def delete(self, delete_all=False, namespace=None):
        self.deletes.append((delete_all, namespace))


class FakePineconeClient:
    def __init__(self, index):
        self.index = index

    def Index(self, name):
        return self.index


def test_pinecone_groups_multitenant_upsert_and_query(monkeypatch):
    from vectordb_bench.backend.clients.pinecone import pinecone as pinecone_module
    from vectordb_bench.backend.clients.pinecone.pinecone import Pinecone

    fake_index = FakePineconeIndex()
    monkeypatch.setattr(
        pinecone_module.pinecone,
        "Pinecone",
        lambda api_key: FakePineconeClient(fake_index),
    )

    db = Pinecone(
        dim=2,
        db_config={
            "api_key": "k",
            "index_name": "idx",
            "multitenant_namespace_prefix": "mt_",
        },
        db_case_config=None,
        drop_old=False,
    )
    db.set_multitenant_context(["tenant_0000", "tenant_0001"])

    with db.init():
        count, err = db.insert_embeddings(
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            metadata=[0, 1, 2],
            tenant_labels_data=["tenant_0000", "tenant_0001", "tenant_0000"],
        )
        result = db.search_embedding([1.0, 0.0], k=1, tenant="tenant_0001")

    assert count == 3
    assert err is None
    assert result == [11]
    assert fake_index.upserts[0][1] == "mt_tenant_0000"
    assert fake_index.upserts[1][1] == "mt_tenant_0001"
    assert fake_index.queries[-1]["namespace"] == "mt_tenant_0001"
