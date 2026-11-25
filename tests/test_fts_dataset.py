import math
from dataclasses import dataclass
from typing import ClassVar

import pytest

from vectordb_bench.backend.dataset import (
    FtsDataset,
    FtsDatasetManager,
    FtsDatasetWithSizeType,
    FtsFilterIdPermutation,
    HotpotQAFts,
    HotpotQATranslator,
    MSMarcoFts,
    MSMarcoTranslator,
    SizeLabel,
)
from vectordb_bench.backend.filter import NewIntFilter, non_filter


@dataclass
class Query:
    query_id: str
    text: str


@dataclass
class Doc:
    doc_id: str
    text: str
    title: str = ""


@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: int


class FakeDataset:
    def __init__(self):
        self.queries = [Query("q1", "alpha query"), Query("q2", "beta query")]
        self.docs = [
            Doc("d1", "alpha document", "A"),
            Doc("d2", "beta document", "B"),
            Doc("d3", "gamma document", "C"),
            Doc("d4", "delta document", "D"),
        ]
        self.qrels = [Qrel("q1", "d3", 1), Qrel("q1", "d2", 0), Qrel("q2", "d1", 2)]

    def queries_iter(self):
        yield from self.queries

    def docs_iter(self):
        yield from self.docs

    def qrels_iter(self):
        yield from self.qrels


class FakeDatasetWithMissingQrel(FakeDataset):
    def __init__(self):
        super().__init__()
        self.qrels = [Qrel("q1", "missing", 1)]


class FakeDatasetWithLowPermutationQrels(FakeDataset):
    def __init__(self):
        super().__init__()
        self.qrels = [Qrel("q1", "d3", 1), Qrel("q2", "d4", 2)]


class FakeDatasetWithLateQrel(FakeDataset):
    def __init__(self):
        super().__init__()
        self.qrels = [Qrel("q1", "d4", 1)]


def make_tiny_msmarco_manager(size: int = 3) -> FtsDatasetManager:
    small_label = MSMarcoFts._size_label[100_000]

    class TinyMSMarcoFts(MSMarcoFts):
        _size_label: ClassVar[dict[int, SizeLabel]] = {
            **MSMarcoFts._size_label,
            size: small_label._replace(size=size),
        }

    return FtsDatasetManager(data=TinyMSMarcoFts(size=size))


def test_msmarco_translator_uses_string_ids_and_qrels():
    translator = MSMarcoTranslator()
    dataset = FakeDataset()

    query = translator.translate_query(Query("10", "hello"))
    document = translator.translate_document(Doc("20", "hello\tworld\nagain"))
    ground_truth = translator.load_ground_truth(dataset)

    assert query.query_id == "10"
    assert document.doc_id == "20"
    assert document.text == "hello world again"
    assert ground_truth == {"q1": {"d3": 1}, "q2": {"d1": 2}}


def test_hotpotqa_translator_combines_title_and_text_and_uses_qrels():
    translator = HotpotQATranslator()
    dataset = FakeDataset()

    document = translator.translate_document(Doc("doc-a", "body", "Title"))
    ground_truth = translator.load_ground_truth(dataset)

    assert document.doc_id == "doc-a"
    assert document.text == "Title body"
    assert ground_truth == {"q1": {"d3": 1}, "q2": {"d1": 2}}


def test_fts_filter_id_permutation_has_stable_sequence():
    permutation = FtsFilterIdPermutation.for_size(8)

    assert permutation.algorithm == "affine_permutation_v1"
    assert permutation.multiplier == 5
    assert permutation.offset == 3
    assert [permutation.map(i) for i in range(8)] == [3, 0, 5, 2, 7, 4, 1, 6]
    assert permutation == FtsFilterIdPermutation.for_size(8)


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 10, 97, 100, 1_000, 100_000, 1_000_000, 5_233_329, 8_841_823])
def test_fts_filter_id_permutation_is_bijective(size: int):
    permutation = FtsFilterIdPermutation.for_size(size)

    assert math.gcd(permutation.multiplier, size) == 1
    assert 0 <= permutation.offset < size
    if size <= 1_000:
        assert sorted(permutation.map(i) for i in range(size)) == list(range(size))


@pytest.mark.parametrize(("size", "filter_rate"), [(8, 0.5), (100, 0.99), (101, 0.75)])
def test_fts_filter_id_permutation_preserves_exact_selectivity(size: int, filter_rate: float):
    permutation = FtsFilterIdPermutation.for_size(size)
    filter_value = int(size * filter_rate)

    matched = sum(permutation.map(i) >= filter_value for i in range(size))

    assert matched == size - filter_value


def test_fts_filter_id_permutation_scatters_matches_across_corpus_order():
    permutation = FtsFilterIdPermutation.for_size(100)

    matching_ordinals = [ordinal for ordinal in range(100) if permutation.map(ordinal) >= 90]

    assert matching_ordinals != list(range(90, 100))
    assert len({ordinal // 10 for ordinal in matching_ordinals}) >= 6


def test_fts_filter_id_permutation_validates_bounds():
    with pytest.raises(ValueError, match="size must be positive"):
        FtsFilterIdPermutation.for_size(0)

    permutation = FtsFilterIdPermutation.for_size(3)
    with pytest.raises(ValueError, match="ordinal must be in"):
        permutation.map(3)


def test_fts_iterator_preserves_qrel_docs_before_filler():
    manager = make_tiny_msmarco_manager()
    manager._ir_dataset = FakeDataset()
    manager.required_doc_ids = {"d4"}
    manager.selected_doc_ids = manager._build_selected_doc_ids()
    manager.filter_stats = {"filter_type": "NumGE"}

    assert manager.selected_doc_ids == {"d1", "d2", "d4"}

    batches = list(manager)
    docs = [(doc.doc_id, doc.filter_id) for batch in batches for doc in batch]

    assert len(docs) == 3
    assert ("d4", 1) in docs
    assert all(not doc_id.isdecimal() for doc_id, _ in docs)
    assert [filter_id for _, filter_id in docs] == [2, 0, 1]


def test_fts_unfiltered_iterator_omits_filter_ids():
    manager = make_tiny_msmarco_manager()
    manager._ir_dataset = FakeDataset()
    manager.selected_doc_ids = {"d1", "d2", "d3"}

    docs = [doc for batch in manager for doc in batch]

    assert [doc.filter_id for doc in docs] == [None, None, None]


def test_fts_prepare_non_filter_materializes_qrel_preserving_corpus(monkeypatch: pytest.MonkeyPatch):
    manager = make_tiny_msmarco_manager(size=3)
    monkeypatch.setattr(manager._translator, "load", FakeDatasetWithLateQrel)

    assert manager.prepare(source=None, filters=non_filter)
    monkeypatch.setattr(
        manager._translator,
        "iter_documents",
        lambda dataset: pytest.fail("timed insertion must use the prepared corpus"),
    )
    docs = [doc for batch in manager for doc in batch]

    assert manager.selected_doc_ids is None
    assert [doc.doc_id for doc in docs] == ["d1", "d2", "d4"]
    assert [doc.filter_id for doc in docs] == [None, None, None]
    assert manager.recall_queries_data == manager.queries_data
    assert manager.recall_gt_data == manager.gt_data


def test_fts_qrel_filter_ids_match_sparse_emitted_documents_when_one_is_skipped():
    class UnassignableDocument:
        doc_id = "d3"
        text = "malformed"

        @property
        def filter_id(self) -> int | None:
            return None

        @filter_id.setter
        def filter_id(self, value: int | None) -> None:  # noqa: ARG002
            raise ValueError("cannot assign filter_id")

    class SparseTranslator:
        def iter_documents(self, dataset):  # noqa: ARG002
            yield Doc("d1", "one")
            yield Doc("d2", "two")
            yield UnassignableDocument()
            yield Doc("d4", "four")
            yield Doc("d5", "five")

    manager = make_tiny_msmarco_manager(size=3)
    manager._ir_dataset = object()
    manager._translator = SparseTranslator()
    manager.required_doc_ids = {"d1", "d5"}
    manager.selected_doc_ids = {"d1", "d3", "d5"}
    manager.filter_stats = {"filter_type": "NumGE"}

    qrel_filter_ids = manager._build_qrel_filter_ids()
    emitted_filter_ids = {doc.doc_id: doc.filter_id for batch in manager for doc in batch}

    assert emitted_filter_ids == {"d1": 2, "d5": 0}
    assert qrel_filter_ids == emitted_filter_ids


def test_fts_prepare_integer_filter_derives_filtered_qrels(monkeypatch: pytest.MonkeyPatch):
    manager = make_tiny_msmarco_manager(size=4)
    monkeypatch.setattr(manager._translator, "load", FakeDataset)

    filters = NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=2)
    assert manager.prepare(source=None, filters=filters)
    monkeypatch.setattr(
        manager._translator,
        "iter_documents",
        lambda dataset: pytest.fail("timed insertion must use the prepared corpus"),
    )
    emitted_filter_ids = {doc.doc_id: doc.filter_id for batch in manager for doc in batch}

    assert [query.query_id for query in manager.queries_data] == ["q1", "q2"]
    assert manager.gt_data == [{"d3": 1}, {"d1": 2}]
    assert [query.query_id for query in manager.recall_queries_data] == ["q2"]
    assert manager.recall_gt_data == [{"d1": 2}]
    assert manager.recall_skipped is False
    assert manager.recall_skip_reason is None
    assert manager.qrel_filter_ids == {"d1": 3, "d3": 1}
    assert emitted_filter_ids == {"d1": 3, "d2": 2, "d3": 1, "d4": 0}
    assert manager.filter_stats == {
        "filter_type": "NumGE",
        "filter_field": "filter_id",
        "filter_value": 2,
        "filter_rate": 0.5,
        "filter_id_distribution": "affine_permutation_v1",
        "filter_id_multiplier": 3,
        "filter_id_offset": 3,
        "matched_doc_count": 2,
        "matched_doc_ratio": 0.5,
        "original_query_count": 2,
        "filtered_query_count": 1,
        "filtered_query_ratio": 0.5,
        "original_relevant_doc_count": 2,
        "filtered_relevant_doc_count": 1,
    }


def test_fts_prepare_integer_filter_skips_empty_filtered_qrels(monkeypatch: pytest.MonkeyPatch):
    manager = make_tiny_msmarco_manager(size=4)
    monkeypatch.setattr(manager._translator, "load", FakeDatasetWithLowPermutationQrels)

    filters = NewIntFilter(filter_rate=0.75, int_field="filter_id", int_value=3)
    assert manager.prepare(source=None, filters=filters)

    assert [query.query_id for query in manager.queries_data] == ["q1", "q2"]
    assert manager.gt_data == [{"d3": 1}, {"d4": 2}]
    assert manager.recall_queries_data == []
    assert manager.recall_gt_data == []
    assert manager.recall_skipped is True
    assert manager.recall_skip_reason == "no_positive_qrels_after_filter"
    assert manager.filter_stats["filtered_query_count"] == 0
    assert manager.filter_stats["filtered_relevant_doc_count"] == 0


def test_fts_prepare_integer_filter_requires_filter_id_field(monkeypatch: pytest.MonkeyPatch):
    manager = make_tiny_msmarco_manager(size=4)
    monkeypatch.setattr(manager._translator, "load", FakeDataset)

    filters = NewIntFilter(filter_rate=0.5, int_field="id", int_value=2)
    with pytest.raises(ValueError, match="int_field='filter_id'"):
        manager.prepare(source=None, filters=filters)


def test_fts_cap_rejects_required_qrel_docs_missing_from_corpus():
    manager = make_tiny_msmarco_manager()
    manager._ir_dataset = FakeDataset()
    manager.required_doc_ids = {"missing"}

    with pytest.raises(ValueError, match="missing from corpus"):
        manager._build_selected_doc_ids()


def test_fts_prepare_propagates_missing_required_qrel_docs(monkeypatch: pytest.MonkeyPatch):
    manager = FtsDatasetManager(data=MSMarcoFts(size=100_000))
    monkeypatch.setattr(manager._translator, "load", FakeDatasetWithMissingQrel)
    filters = NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=50_000)

    with pytest.raises(ValueError, match="missing from corpus"):
        manager.prepare(source=None, filters=filters)


def test_fts_dataset_size_registry():
    assert FtsDataset.MSMARCO.manager(100_000).data.full_name == "MS MARCO FTS (SMALL)"
    assert FtsDataset.MSMARCO.manager(1_000_000).data.full_name == "MS MARCO FTS (MEDIUM)"
    assert FtsDataset.HOTPOTQA.manager(5_233_329).data.full_name == "HotpotQA FTS (LARGE)"
    assert FtsDatasetWithSizeType.MSMarcoSmall.get_manager().data.size == 100_000
    assert FtsDatasetWithSizeType.HotpotQAMedium.get_manager().data.size == 1_000_000
    with pytest.raises(ValueError, match="not supported"):
        FtsDataset.MSMARCO.manager(3)


def test_cap_smaller_than_required_qrels_is_invalid():
    manager = FtsDatasetManager(data=HotpotQAFts(size=100_000))
    manager.qrels_data = {"q1": {"a": 1, "b": 1, "c": 1}}

    with pytest.raises(ValueError, match="requires 3 qrel documents"):
        manager._validate_cap(required_doc_ids={"a", "b", "c"}, target_size=2)
