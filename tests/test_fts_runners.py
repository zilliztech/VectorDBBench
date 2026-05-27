from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from vectordb_bench.backend.workload import WorkloadKind


class FakeDB:
    name = "FakeDB"

    def __init__(self):
        self.calls = []

    def search_embedding(self, query, k=100):
        self.calls.append(("vector", query, k))
        return ["1"]

    def search_documents(self, query, k=100):
        self.calls.append(("fts", query, k))
        return ["doc-1"]


def test_serial_runner_uses_explicit_vector_workload():
    db = FakeDB()
    runner = SerialSearchRunner(db=db, test_data=[[0.1]], ground_truth=[["1"]], k=3, workload_kind=WorkloadKind.VECTOR)

    assert runner._get_db_search_res([0.1]) == ["1"]
    assert db.calls == [("vector", [0.1], 3)]


def test_serial_runner_uses_explicit_fts_workload():
    db = FakeDB()
    runner = SerialSearchRunner(
        db=db,
        test_data=["alpha"],
        ground_truth=[["doc-1"]],
        k=5,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    assert runner._get_db_search_res("alpha") == ["doc-1"]
    assert db.calls == [("fts", "alpha", 5)]
    assert runner._use_fts_metrics is True


def test_mp_runner_uses_explicit_fts_workload():
    db = FakeDB()
    runner = MultiProcessingSearchRunner(
        db=db,
        test_data=["alpha"],
        k=5,
        workload_kind=WorkloadKind.FULL_TEXT_BM25,
    )

    assert runner._search_func("alpha", 5) == ["doc-1"]
