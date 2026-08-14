"""Unit tests for RatedMultiThreadingInsertRunner.send_insert_task routing.

These exercise the thread-safety routing without a live database: a thread-safe
client inserts through the shared object, while a non-thread-safe client is routed
through copy_for_thread() + init() so each worker owns its own connection.
"""

from contextlib import contextmanager
from copy import deepcopy

from vectordb_bench.backend.clients.api import VectorDB
from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner


class FakeDB(VectorDB):
    def __init__(self, thread_safe: bool = True, name: str = "Fake"):
        self.thread_safe = thread_safe
        self.name = name
        self.init_calls = 0
        self.inserted: list[int] = []

    @contextmanager
    def init(self):
        self.init_calls += 1
        yield

    def insert_embeddings(self, embeddings: list, metadata: list, **kwargs):
        self.inserted.append(len(embeddings))
        return len(embeddings), None

    def search_embedding(self, *args, **kwargs):
        return []

    def optimize(self, data_size: int | None = None):
        return


class CapturingNonThreadSafeDB(FakeDB):
    """Records the per-thread copy so the test can assert against it."""

    def __init__(self):
        super().__init__(thread_safe=False, name="Capturing")
        self.thread_copy: CapturingNonThreadSafeDB | None = None

    def copy_for_thread(self) -> "VectorDB":
        c = deepcopy(self)
        self.thread_copy = c
        return c


def _runner(db: VectorDB):
    return RatedMultiThreadingInsertRunner(rate=10, db=db, dataset_iter=None)


def test_thread_safe_client_uses_shared_object():
    db = FakeDB(thread_safe=True)
    _runner(db).send_insert_task(db, [[0.1], [0.2]], ["a", "b"])

    # Shared client inserts directly, no per-thread copy/init from the runner.
    assert db.inserted == [2]
    assert db.init_calls == 0


def test_non_thread_safe_client_routes_through_copy():
    db = CapturingNonThreadSafeDB()
    _runner(db).send_insert_task(db, [[0.1], [0.2]], ["a", "b"])

    copy = db.thread_copy
    assert copy is not None
    assert copy is not db
    # Insert + init happen on the thread-local copy, never the parent.
    assert copy.init_calls == 1
    assert copy.inserted == [2]
    assert db.inserted == []
    assert db.init_calls == 0


def test_default_copy_for_thread_is_a_distinct_deep_copy():
    db = FakeDB(thread_safe=False)
    copy = db.copy_for_thread()

    assert copy is not db
    assert isinstance(copy, FakeDB)
    copy.inserted.append(1)
    assert db.inserted == []
