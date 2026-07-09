from dataclasses import dataclass
from typing import ClassVar

import pytest

from vectordb_bench.backend.dataset import (
    FtsDataset,
    FtsDatasetManager,
    FtsDatasetWithSizeType,
    HotpotQAFts,
    HotpotQATranslator,
    MSMarcoFts,
    MSMarcoTranslator,
    SizeLabel,
)
from vectordb_bench.backend.filter import NewIntFilter


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


def test_fts_iterator_preserves_qrel_docs_before_filler():
    manager = make_tiny_msmarco_manager()
    manager._ir_dataset = FakeDataset()
    manager.required_doc_ids = {"d4"}
    manager.selected_doc_ids = manager._build_selected_doc_ids()

    assert manager.selected_doc_ids == {"d1", "d2", "d4"}

    batches = list(manager)
    docs = [(doc.doc_id, doc.filter_id) for batch in batches for doc in batch]

    assert len(docs) == 3
    assert ("d4", 2) in docs
    assert all(not doc_id.isdecimal() for doc_id, _ in docs)
    assert [filter_id for _, filter_id in docs] == [0, 1, 2]


def test_fts_prepare_integer_filter_derives_filtered_qrels(monkeypatch: pytest.MonkeyPatch):
    manager = make_tiny_msmarco_manager(size=4)
    monkeypatch.setattr(manager._translator, "load", FakeDataset)

    filters = NewIntFilter(filter_rate=0.5, int_field="filter_id", int_value=2)
    assert manager.prepare(source=None, filters=filters)

    assert [query.query_id for query in manager.queries_data] == ["q1", "q2"]
    assert manager.gt_data == [{"d3": 1}, {"d1": 2}]
    assert [query.query_id for query in manager.recall_queries_data] == ["q1"]
    assert manager.recall_gt_data == [{"d3": 1}]
    assert manager.recall_skipped is False
    assert manager.recall_skip_reason is None
    assert manager.qrel_filter_ids == {"d1": 0, "d3": 2}
    assert manager.filter_stats == {
        "filter_type": "NumGE",
        "filter_field": "filter_id",
        "filter_value": 2,
        "filter_rate": 0.5,
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
    monkeypatch.setattr(manager._translator, "load", FakeDataset)

    filters = NewIntFilter(filter_rate=0.75, int_field="filter_id", int_value=3)
    assert manager.prepare(source=None, filters=filters)

    assert [query.query_id for query in manager.queries_data] == ["q1", "q2"]
    assert manager.gt_data == [{"d3": 1}, {"d1": 2}]
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

    with pytest.raises(ValueError, match="missing from corpus"):
        manager.prepare(source=None)


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
