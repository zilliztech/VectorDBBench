import threading
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


class FailingPineconeIndex(FakePineconeIndex):
    def upsert(self, vectors, namespace=None):
        self.upserts.append((vectors, namespace))
        if namespace == "mt_tenant_0001":
            raise RuntimeError("tenant upsert failed")
        return SimpleNamespace(_response_info={"raw_headers": {"x-pinecone-request-lsn": "7"}})


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


def test_pinecone_multitenant_upsert_preserves_scalar_payload_labels(monkeypatch):
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
        with_scalar_labels=True,
    )

    with db.init():
        count, err = db.insert_embeddings(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadata=[0, 1],
            labels_data=["label_a", "label_b"],
            tenant_labels_data=["tenant_0000", "tenant_0001"],
        )

    assert count == 2
    assert err is None
    assert fake_index.upserts[0][0][0][2]["label"] == "label_a"
    assert fake_index.upserts[1][0][0][2]["label"] == "label_b"


def test_pinecone_multitenant_partial_insert_failure_is_explicit(monkeypatch):
    from vectordb_bench.backend.clients.pinecone import pinecone as pinecone_module
    from vectordb_bench.backend.clients.pinecone.pinecone import Pinecone

    fake_index = FailingPineconeIndex()
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

    with db.init():
        count, err = db.insert_embeddings(
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            metadata=[0, 1, 2],
            tenant_labels_data=["tenant_0000", "tenant_0001", "tenant_0000"],
        )

    assert count == 2
    assert getattr(err, "non_retryable", False) is True
    assert getattr(err, "inserted_count") == 2
    assert getattr(err, "successful_tenants") == {"tenant_0000": 2}
    assert getattr(err, "failed_tenant") == "tenant_0001"
    assert getattr(err, "failed_tenant_count") == 1
    assert db._multitenant_insert_counts == {"tenant_0000": 2}


def test_pinecone_multitenant_insert_counts_are_thread_safe(monkeypatch):
    from vectordb_bench.backend.clients.pinecone import pinecone as pinecone_module
    from vectordb_bench.backend.clients.pinecone.pinecone import Pinecone

    class RacingCounts(dict):
        def __init__(self, lock_getter):
            super().__init__()
            self.barrier = threading.Barrier(2)
            self.lock_getter = lock_getter

        def get(self, key, default=None):
            value = super().get(key, default)
            lock = self.lock_getter()
            if key == "tenant_0000" and (lock is None or not lock.locked()):
                self.barrier.wait(timeout=5)
            return value

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
    db._multitenant_insert_counts = RacingCounts(lambda: getattr(db, "_multitenant_insert_counts_lock", None))
    results = []

    def insert_one(row_id):
        results.append(
            db.insert_embeddings(
                embeddings=[[float(row_id), 0.0]],
                metadata=[row_id],
                tenant_labels_data=["tenant_0000"],
            )
        )

    with db.init():
        threads = [threading.Thread(target=insert_one, args=(row_id,)) for row_id in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(fake_index.upserts) == 2
    assert sorted(results) == [(1, None), (1, None)]
    assert dict(db._multitenant_insert_counts) == {"tenant_0000": 2}
