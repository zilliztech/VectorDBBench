from types import TracebackType
from typing import Any

import pytest

from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from vectordb_bench.backend.workload import WorkloadKind
from vectordb_bench.metric import calc_mrr_fts, calc_ndcg_fts, calc_recall_fts


class FakeSearchDB:
    name = "FakeSearchDB"

    def init(self):
        return self

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    def prepare_filter(self, filters: Any) -> None:
        return None

    def supports_document_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile == PayloadProfile.IDS_ONLY

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return True

    def search_documents(self, query: str, k: int) -> list[str]:
        assert query == "query"
        assert k == 3
        return ["missing", "d2", "d1"]


def test_semantic_fts_metrics_use_relevance_grades():
    qrels = {"d1": 3, "d2": 1}
    got = ["missing", "d2", "d1"]

    assert calc_recall_fts(2, qrels, got) == 0.5
    assert calc_recall_fts(3, qrels, got) == 1.0
    assert calc_mrr_fts(3, qrels, got) == 0.5
    assert calc_ndcg_fts(3, qrels, got) == pytest.approx(0.5869, abs=1e-4)


def test_serial_runner_returns_semantic_fts_metrics():
    runner = SerialSearchRunner(
        db=FakeSearchDB(),
        test_data=["query"],
        ground_truth=[{"d1": 3, "d2": 1}],
        k=3,
        workload_kind=WorkloadKind.FULL_TEXT,
    )

    recall, ndcg, mrr, p99, p95 = runner.search((runner.test_data, runner.ground_truth))

    assert recall == 1.0
    assert ndcg == 0.5869
    assert mrr == 0.5
    assert p99 >= 0
    assert p95 >= 0
