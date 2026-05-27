# FTS BM25 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement IDs-only native BM25 full-text retrieval benchmarks for Milvus, ElasticCloud, Vespa, and Turbopuffer over MS MARCO and HotpotQA with qrels-based quality metrics.

**Architecture:** Keep FTS dataset and backend text-search semantics separate, but run FTS through the shared VectorDBBench performance lifecycle. Add explicit workload/capability boundaries so BM25 is not encoded as the opposite of vector search and future hybrid dense+BM25 remains viable.

**Tech Stack:** Python, Pydantic v1, ir-datasets, pymilvus, elasticsearch-py, pyvespa, turbopuffer SDK, pytest, Streamlit config metadata.

---

## File Structure

- Create `vectordb_bench/backend/workload.py`: defines `WorkloadKind`.
- Modify `vectordb_bench/backend/clients/api.py`: adds optional full-text capability methods and keeps vector methods unchanged.
- Modify `vectordb_bench/backend/dataset.py`: fixes FTS IDs/qrels/caps, adds HotpotQA and full size registry.
- Modify `vectordb_bench/backend/cases.py`: makes the FTS case parameterized by `FtsDatasetWithSizeType`.
- Modify `vectordb_bench/backend/runner/serial_runner.py`: uses explicit workload kind for serial search and keeps `SerialFtsInsertRunner` separate.
- Modify `vectordb_bench/backend/runner/mp_runner.py`: uses explicit workload kind for concurrent search.
- Modify `vectordb_bench/backend/task_runner.py`: folds FTS into the shared performance lifecycle and uses mode-specific load/search adapters.
- Modify `vectordb_bench/backend/assembler.py`: gates FTS support before task execution and preserves `IR_DATASETS` source selection.
- Modify `vectordb_bench/backend/clients/milvus/config.py`: fixes analyzer param wiring and keeps BM25 config.
- Modify `vectordb_bench/backend/clients/milvus/milvus.py`: supports string doc IDs and explicit FTS workload setup.
- Modify `vectordb_bench/backend/clients/elastic_cloud/config.py`: adds `ElasticCloudFtsConfig`.
- Modify `vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py`: implements BM25 document insert/search.
- Modify `vectordb_bench/backend/clients/vespa/config.py`: adds `VespaFtsConfig`.
- Modify `vectordb_bench/backend/clients/vespa/vespa.py`: implements BM25 document insert/search.
- Modify `vectordb_bench/backend/clients/turbopuffer/config.py`: adds `TurboPufferFtsConfig`.
- Modify `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py`: implements BM25 document insert/search.
- Modify `vectordb_bench/backend/clients/__init__.py`: maps `IndexType.FTS_AUTOINDEX` to FTS configs for supported DBs.
- Modify `vectordb_bench/frontend/config/dbCaseConfigs.py`: exposes Small/Medium FTS cases by default and Large as advanced/opt-in, capability-gated to Milvus, ElasticCloud, Vespa, Turbopuffer.
- Modify `vectordb_bench/cli/cli.py`: lets CLI select FTS dataset tiers for supported backends.
- Modify `install/requirements_py3.11.txt`: adds `ir_datasets`.
- Create tests:
  - `tests/test_fts_dataset.py`
  - `tests/test_fts_cases.py`
  - `tests/test_fts_runners.py`
  - `tests/test_fts_backend_capability.py`
  - `tests/test_fts_elastic_cloud.py`
  - `tests/test_fts_milvus.py`
  - `tests/test_fts_vespa.py`
  - `tests/test_fts_turbopuffer.py`

## Task 1: Workload And Capability Boundary

**Files:**
- Create: `vectordb_bench/backend/workload.py`
- Modify: `vectordb_bench/backend/clients/api.py`
- Test: `tests/test_fts_backend_capability.py`

- [ ] **Step 1: Write failing capability tests**

Create `tests/test_fts_backend_capability.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_backend_capability.py -q
```

Expected: FAIL because `vectordb_bench.backend.workload` does not exist and `VectorDB` has no optional FTS methods.

- [ ] **Step 3: Add workload enum**

Create `vectordb_bench/backend/workload.py`:

```python
from enum import StrEnum


class WorkloadKind(StrEnum):
    VECTOR = "vector"
    FULL_TEXT_BM25 = "full_text_bm25"
    HYBRID_DENSE_BM25 = "hybrid_dense_bm25"
```

- [ ] **Step 4: Add optional FTS methods to base client**

In `vectordb_bench/backend/clients/api.py`, add these methods inside `class VectorDB` after `need_normalize_cosine()`:

```python
    @classmethod
    def supports_full_text_search(cls) -> bool:
        return False

    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        msg = f"{self.name or self.__class__.__name__} does not support full-text document insert"
        raise NotImplementedError(msg)

    def search_documents(
        self,
        query: str,
        k: int = 100,
        **kwargs,
    ) -> list[str]:
        msg = f"{self.name or self.__class__.__name__} does not support full-text document search"
        raise NotImplementedError(msg)
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
pytest tests/test_fts_backend_capability.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/workload.py vectordb_bench/backend/clients/api.py tests/test_fts_backend_capability.py
git commit -m "feat: add fts workload capability boundary"
```

## Task 2: FTS Dataset Correctness

**Files:**
- Modify: `vectordb_bench/backend/dataset.py`
- Test: `tests/test_fts_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/test_fts_dataset.py`:

```python
from dataclasses import dataclass

import pytest

from vectordb_bench.backend.dataset import (
    FtsDataset,
    FtsDatasetManager,
    FtsDatasetWithSizeType,
    HotpotQAFts,
    HotpotQATranslator,
    MSMarcoFts,
    MSMarcoTranslator,
)


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


def test_msmarco_translator_uses_string_ids_and_qrels():
    translator = MSMarcoTranslator()
    dataset = FakeDataset()

    query = translator.translate_query(Query("10", "hello"))
    document = translator.translate_document(Doc("20", "hello\tworld\nagain"))
    ground_truth = translator.load_ground_truth(dataset)

    assert query.query_id == "10"
    assert document.doc_id == "20"
    assert document.text == "hello world again"
    assert ground_truth == {"q1": ["d3"], "q2": ["d1"]}


def test_hotpotqa_translator_combines_title_and_text_and_uses_qrels():
    translator = HotpotQATranslator()
    dataset = FakeDataset()

    document = translator.translate_document(Doc("doc-a", "body", "Title"))
    ground_truth = translator.load_ground_truth(dataset)

    assert document.doc_id == "doc-a"
    assert document.text == "Title body"
    assert ground_truth == {"q1": ["d3"], "q2": ["d1"]}


def test_fts_iterator_preserves_qrel_docs_before_filler(monkeypatch):
    manager = FtsDatasetManager(data=MSMarcoFts(size=3))
    manager._ir_dataset = FakeDataset()
    manager.required_doc_ids = {"d3", "d1"}
    manager.selected_doc_ids = None

    batches = list(manager)
    docs = [doc.doc_id for batch in batches for doc in batch]

    assert len(docs) == 3
    assert set(["d3", "d1"]).issubset(set(docs))


def test_fts_dataset_size_registry():
    assert FtsDataset.MSMARCO.manager(100_000).data.full_name == "MS MARCO FTS (SMALL)"
    assert FtsDataset.MSMARCO.manager(1_000_000).data.full_name == "MS MARCO FTS (MEDIUM)"
    assert FtsDataset.HOTPOTQA.manager(5_233_329).data.full_name == "HotpotQA FTS (LARGE)"
    assert FtsDatasetWithSizeType.MSMarcoSmall.get_manager().data.size == 100_000
    assert FtsDatasetWithSizeType.HotpotQAMedium.get_manager().data.size == 1_000_000


def test_cap_smaller_than_required_qrels_is_invalid():
    manager = FtsDatasetManager(data=HotpotQAFts(size=100_000))
    manager.qrels_data = {"q1": ["a", "b", "c"]}

    with pytest.raises(ValueError, match="requires 3 qrel documents"):
        manager._validate_cap(required_doc_ids={"a", "b", "c"}, target_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_dataset.py -q
```

Expected: FAIL because IDs are integers, HotpotQA does not exist, qrels use `scoreddocs_iter()`, and caps are not enforced.

- [ ] **Step 3: Update FTS dataclasses and ground truth type**

In `vectordb_bench/backend/dataset.py`, replace FTS ID types with strings:

```python
@dataclass
class FtsQuery:
    """Internal representation of an FTS query."""

    query_id: str
    text: str


@dataclass
class FtsDocument:
    """Internal representation of an FTS document."""

    doc_id: str
    text: str


FtsGroundTruth = dict[str, list[str]]
```

- [ ] **Step 4: Add qrels helper and fix MS MARCO translator**

Add this helper near `FtsDatasetTranslator`:

```python
def _load_positive_qrels(dataset: typing.Any) -> FtsGroundTruth:
    gt: FtsGroundTruth = {}
    for qrel in dataset.qrels_iter():
        relevance = int(getattr(qrel, "relevance", 1))
        if relevance <= 0:
            continue
        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        gt.setdefault(query_id, []).append(doc_id)
    return gt
```

Update `MSMarcoTranslator`:

```python
class MSMarcoTranslator(FtsDatasetTranslator):
    """Translator for MS MARCO passage retrieval dataset."""

    @property
    def ir_datasets_name(self) -> str:
        return "msmarco-passage/dev/small"

    def translate_query(self, ir_query: typing.Any) -> FtsQuery:
        return FtsQuery(query_id=str(ir_query.query_id), text=ir_query.text)

    def translate_document(self, ir_doc: typing.Any) -> FtsDocument:
        clean_text = ir_doc.text.replace("\t", " ").replace("\n", " ")
        return FtsDocument(doc_id=str(ir_doc.doc_id), text=clean_text)

    def load_ground_truth(self, dataset: typing.Any) -> FtsGroundTruth:
        return _load_positive_qrels(dataset)
```

- [ ] **Step 5: Add HotpotQA translator and dataset classes**

Add below `MSMarcoTranslator`:

```python
class HotpotQATranslator(FtsDatasetTranslator):
    """Translator for BEIR HotpotQA."""

    @property
    def ir_datasets_name(self) -> str:
        return "beir/hotpotqa/test"

    def translate_query(self, ir_query: typing.Any) -> FtsQuery:
        return FtsQuery(query_id=str(ir_query.query_id), text=ir_query.text)

    def translate_document(self, ir_doc: typing.Any) -> FtsDocument:
        title = getattr(ir_doc, "title", "") or ""
        text = getattr(ir_doc, "text", "") or ""
        clean_text = f"{title} {text}".replace("\t", " ").replace("\n", " ").strip()
        return FtsDocument(doc_id=str(ir_doc.doc_id), text=clean_text)

    def load_ground_truth(self, dataset: typing.Any) -> FtsGroundTruth:
        return _load_positive_qrels(dataset)
```

Add these dataset classes:

```python
class MSMarcoFts(FtsBaseDataset):
    name: str = "MS MARCO"
    with_gt: bool = True
    with_remote_resource: bool = False

    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        8_841_823: SizeLabel(8_841_823, "LARGE", 1),
    }


class HotpotQAFts(FtsBaseDataset):
    name: str = "HotpotQA"
    with_gt: bool = True
    with_remote_resource: bool = False

    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        5_233_329: SizeLabel(5_233_329, "LARGE", 1),
    }
```

- [ ] **Step 6: Add deterministic cap construction to manager and iterator**

Update `FtsDatasetManager` fields and helpers:

```python
    required_doc_ids: set[str] | None = None
    selected_doc_ids: set[str] | None = None

    def _validate_cap(self, required_doc_ids: set[str], target_size: int) -> None:
        if len(required_doc_ids) > target_size:
            msg = f"{self.data.full_name} cap {target_size} requires {len(required_doc_ids)} qrel documents"
            raise ValueError(msg)

    def _build_required_doc_ids(self) -> set[str]:
        if not self.qrels_data:
            return set()
        return {doc_id for doc_ids in self.qrels_data.values() for doc_id in doc_ids}

    def _build_selected_doc_ids(self) -> set[str] | None:
        if self._is_large():
            return None
        required_doc_ids = self.required_doc_ids or set()
        self._validate_cap(required_doc_ids, self.data.size)
        selected_doc_ids: set[str] = set(required_doc_ids)
        for ir_doc in self.ir_dataset.docs_iter():
            if len(selected_doc_ids) >= self.data.size:
                break
            selected_doc_ids.add(str(ir_doc.doc_id))
        if len(selected_doc_ids) < self.data.size:
            msg = f"{self.data.full_name} only selected {len(selected_doc_ids)} documents for cap {self.data.size}"
            raise ValueError(msg)
        return selected_doc_ids

    def _is_large(self) -> bool:
        return self.data.label == "LARGE"
```

In `prepare()`, after loading `qrels_data`, set:

```python
                self.required_doc_ids = self._build_required_doc_ids()
                self.selected_doc_ids = self._build_selected_doc_ids()
```

Update the inner batch loop in `FtsDocumentIterator.__next__()` to stop at `self._ds.data.size` and filter by `selected_doc_ids` for capped tiers:

```python
            for _ in range(self._batch_size):
                if self._doc_count >= self._ds.data.size:
                    self._finished = True
                    if batch:
                        return batch
                    raise StopIteration
                try:
                    while True:
                        doc = next(self._docs_iter)
                        selected_doc_ids = self._ds.selected_doc_ids
                        if selected_doc_ids is None or doc.doc_id in selected_doc_ids:
                            break
                    batch.append(doc)
                    self._doc_count += 1
                except StopIteration:
                    self._finished = True
                    if batch:
                        return batch
                    raise
                except Exception as e:
                    log.debug(f"Skipping malformed document: {e}")
                    continue
```

- [ ] **Step 7: Update FTS dataset registries**

Replace `FtsDataset` and `FtsDatasetWithSizeType` bodies with:

```python
class FtsDataset(Enum):
    MSMARCO = MSMarcoFts
    HOTPOTQA = HotpotQAFts

    def get(self, size: int) -> FtsBaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> FtsDatasetManager:
        return FtsDatasetManager(data=self.get(size))


class FtsDatasetWithSizeType(Enum):
    MSMarcoSmall = "MS MARCO Small (100K documents)"
    MSMarcoMedium = "MS MARCO Medium (1M documents)"
    MSMarcoLarge = "MS MARCO Large (8.8M documents)"
    HotpotQASmall = "HotpotQA Small (100K documents)"
    HotpotQAMedium = "HotpotQA Medium (1M documents)"
    HotpotQALarge = "HotpotQA Large (5.2M documents)"

    def get_manager(self) -> FtsDatasetManager:
        return {
            FtsDatasetWithSizeType.MSMarcoSmall: FtsDataset.MSMARCO.manager(100_000),
            FtsDatasetWithSizeType.MSMarcoMedium: FtsDataset.MSMARCO.manager(1_000_000),
            FtsDatasetWithSizeType.MSMarcoLarge: FtsDataset.MSMARCO.manager(8_841_823),
            FtsDatasetWithSizeType.HotpotQASmall: FtsDataset.HOTPOTQA.manager(100_000),
            FtsDatasetWithSizeType.HotpotQAMedium: FtsDataset.HOTPOTQA.manager(1_000_000),
            FtsDatasetWithSizeType.HotpotQALarge: FtsDataset.HOTPOTQA.manager(5_233_329),
        }[self]

    def get_load_timeout(self) -> float:
        if self in {FtsDatasetWithSizeType.MSMarcoSmall, FtsDatasetWithSizeType.HotpotQASmall}:
            return config.LOAD_TIMEOUT_768D_100K
        return config.LOAD_TIMEOUT_DEFAULT

    def get_optimize_timeout(self) -> float:
        if self in {FtsDatasetWithSizeType.MSMarcoSmall, FtsDatasetWithSizeType.HotpotQASmall}:
            return config.OPTIMIZE_TIMEOUT_768D_100K
        return config.OPTIMIZE_TIMEOUT_DEFAULT

    @property
    def is_advanced(self) -> bool:
        return self in {FtsDatasetWithSizeType.MSMarcoLarge, FtsDatasetWithSizeType.HotpotQALarge}
```

- [ ] **Step 8: Run dataset tests**

Run:

```bash
pytest tests/test_fts_dataset.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add vectordb_bench/backend/dataset.py tests/test_fts_dataset.py
git commit -m "fix: correct fts dataset qrels and caps"
```

## Task 3: Parameterized FTS Cases

**Files:**
- Modify: `vectordb_bench/backend/cases.py`
- Test: `tests/test_fts_cases.py`

- [ ] **Step 1: Write failing case tests**

Create `tests/test_fts_cases.py`:

```python
from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType
from vectordb_bench.models import CaseConfig


def test_fts_case_defaults_to_msmarco_small():
    case = CaseConfig(case_id=CaseType.FTSmsmarcoPerformance).case

    assert case.label == CaseLabel.FullTextSearchPerformance
    assert case.dataset_with_size_type == FtsDatasetWithSizeType.MSMarcoSmall
    assert case.dataset.data.size == 100_000
    assert "MS MARCO Small" in case.name


def test_fts_case_accepts_hotpotqa_medium():
    case = CaseConfig(
        case_id=CaseType.FTSmsmarcoPerformance,
        custom_case={"dataset_with_size_type": FtsDatasetWithSizeType.HotpotQAMedium.value},
    ).case

    assert case.dataset_with_size_type == FtsDatasetWithSizeType.HotpotQAMedium
    assert case.dataset.data.name == "HotpotQA"
    assert case.dataset.data.size == 1_000_000
    assert "HotpotQA Medium" in case.name
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_cases.py -q
```

Expected: FAIL because `FTSmsmarcoPerformance` is not parameterized.

- [ ] **Step 3: Import `FtsDatasetWithSizeType` and update FTS case**

In `vectordb_bench/backend/cases.py`, update the dataset import:

```python
from .dataset import (
    CustomDataset,
    Dataset,
    DatasetManager,
    DatasetWithSizeType,
    FtsDataset,
    FtsDatasetManager,
    FtsDatasetWithSizeType,
)
```

Replace `FTSmsmarcoPerformance` with:

```python
class FTSmsmarcoPerformance(FtsPerformanceCase):
    case_id: CaseType = CaseType.FTSmsmarcoPerformance
    dataset_with_size_type: FtsDatasetWithSizeType = FtsDatasetWithSizeType.MSMarcoSmall

    def __init__(
        self,
        dataset_with_size_type: FtsDatasetWithSizeType | str = FtsDatasetWithSizeType.MSMarcoSmall,
        **kwargs,
    ):
        if not isinstance(dataset_with_size_type, FtsDatasetWithSizeType):
            dataset_with_size_type = FtsDatasetWithSizeType(dataset_with_size_type)
        dataset = dataset_with_size_type.get_manager()
        name = f"FTS BM25 Performance - {dataset_with_size_type.value}"
        description = (
            f"This case tests native BM25 full-text search performance on {dataset_with_size_type.value}. "
            "It measures index building time, recall, NDCG, MRR, serial latency, and search QPS."
        )
        super().__init__(
            name=name,
            description=description,
            dataset=dataset,
            dataset_with_size_type=dataset_with_size_type,
            load_timeout=dataset_with_size_type.get_load_timeout(),
            optimize_timeout=dataset_with_size_type.get_optimize_timeout(),
            **kwargs,
        )
```

- [ ] **Step 4: Run case tests**

Run:

```bash
pytest tests/test_fts_cases.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/backend/cases.py tests/test_fts_cases.py
git commit -m "feat: parameterize fts performance cases"
```

## Task 4: Explicit Search Workload In Runners

**Files:**
- Modify: `vectordb_bench/backend/runner/serial_runner.py`
- Modify: `vectordb_bench/backend/runner/mp_runner.py`
- Test: `tests/test_fts_runners.py`

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_fts_runners.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_runners.py -q
```

Expected: FAIL because runners accept `search_fulltext`, not `workload_kind`.

- [ ] **Step 3: Update `SerialSearchRunner` signature and dispatch**

In `vectordb_bench/backend/runner/serial_runner.py`, import:

```python
from vectordb_bench.backend.workload import WorkloadKind
```

Change `SerialSearchRunner.__init__` signature to:

```python
        workload_kind: WorkloadKind = WorkloadKind.VECTOR,
```

Replace the `search_fulltext` autodetection block with:

```python
        self.workload_kind = workload_kind
        if workload_kind == WorkloadKind.FULL_TEXT_BM25:
            self._search_func = self.db.search_documents
            self._use_fts_metrics = True
        elif workload_kind == WorkloadKind.VECTOR:
            self._search_func = self.db.search_embedding
            self._use_fts_metrics = False
        else:
            msg = f"Unsupported search workload: {workload_kind}"
            raise NotImplementedError(msg)
```

Keep `_get_db_search_res()` behavior unchanged and rename its parameter from `emb` to `query`:

```python
    def _get_db_search_res(self, query):
        res = self._search_func(query, self.k)
        return res
```

- [ ] **Step 4: Update `MultiProcessingSearchRunner` signature and dispatch**

In `vectordb_bench/backend/runner/mp_runner.py`, import:

```python
from vectordb_bench.backend.workload import WorkloadKind
```

Change `__init__` signature from `search_fulltext: bool = False` to:

```python
        workload_kind: WorkloadKind = WorkloadKind.VECTOR,
```

Replace dispatch with:

```python
        self.workload_kind = workload_kind
        if workload_kind == WorkloadKind.FULL_TEXT_BM25:
            self._search_func = self.db.search_documents
        elif workload_kind == WorkloadKind.VECTOR:
            self._search_func = self.db.search_embedding
        else:
            msg = f"Unsupported search workload: {workload_kind}"
            raise NotImplementedError(msg)
```

- [ ] **Step 5: Update current call sites**

In `vectordb_bench/backend/task_runner.py`, replace `search_fulltext=True` with:

```python
workload_kind=WorkloadKind.FULL_TEXT_BM25,
```

and pass:

```python
workload_kind=WorkloadKind.VECTOR,
```

at vector `SerialSearchRunner` and `MultiProcessingSearchRunner` call sites where keyword arguments are already being edited in the same block.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_fts_runners.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vectordb_bench/backend/runner/serial_runner.py vectordb_bench/backend/runner/mp_runner.py vectordb_bench/backend/task_runner.py tests/test_fts_runners.py
git commit -m "refactor: use explicit workload kind in search runners"
```

## Task 5: Shared Performance Lifecycle

**Files:**
- Modify: `vectordb_bench/backend/task_runner.py`
- Test: `tests/test_fts_runners.py`

- [ ] **Step 1: Add lifecycle regression test**

Append to `tests/test_fts_runners.py`:

```python
from unittest.mock import Mock

from vectordb_bench.models import TaskStage


def test_fts_and_vector_perf_paths_use_same_orchestration_methods():
    from vectordb_bench.backend.task_runner import CaseRunner

    assert hasattr(CaseRunner, "_run_perf_case")
    assert not hasattr(CaseRunner, "_run_fts_perf_case")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_runners.py::test_fts_and_vector_perf_paths_use_same_orchestration_methods -q
```

Expected: FAIL because `_run_fts_perf_case` still exists.

- [ ] **Step 3: Add workload helpers to `CaseRunner`**

In `vectordb_bench/backend/task_runner.py`, import:

```python
from .workload import WorkloadKind
```

Add methods to `CaseRunner`:

```python
    @property
    def workload_kind(self) -> WorkloadKind:
        if self.ca.label == CaseLabel.FullTextSearchPerformance:
            return WorkloadKind.FULL_TEXT_BM25
        return WorkloadKind.VECTOR

    @property
    def is_fts(self) -> bool:
        return self.workload_kind == WorkloadKind.FULL_TEXT_BM25
```

- [ ] **Step 4: Replace binary `dim = 0` logic with workload-aware dim**

In `init_db()`, replace:

```python
dim = 0 if self.ca.label == CaseLabel.FullTextSearchPerformance else self.ca.dataset.data.dim
```

with:

```python
dim = getattr(self.ca.dataset.data, "dim", 0)
```

This keeps first-pass FTS at dim 0 because FTS datasets have no `dim`, without making `CaseLabel` the permanent schema discriminator.

- [ ] **Step 5: Fold `_run_fts_perf_case()` into `_run_perf_case()`**

In `run()`, replace:

```python
        if self.ca.label == CaseLabel.Performance:
            return self._run_perf_case(drop_old)

        if self.ca.label == CaseLabel.FullTextSearchPerformance:
            return self._run_fts_perf_case(drop_old)
```

with:

```python
        if self.ca.label in {CaseLabel.Performance, CaseLabel.FullTextSearchPerformance}:
            return self._run_perf_case(drop_old)
```

In `_run_perf_case()`, replace `_load_train_data()` call with:

```python
                    _, load_dur = self._load_data()
```

Replace `_init_search_runner()` call with:

```python
                self._init_search_runners()
```

Add:

```python
    @utils.time_it
    def _load_data(self):
        if self.is_fts:
            return self._load_fts_data()
        return self._load_train_data()

    def _init_search_runners(self):
        if self.is_fts:
            return self._init_fts_search_runner()
        return self._init_search_runner()
```

Remove `_run_fts_perf_case()` after `_run_perf_case()` uses the shared load/search dispatch.

- [ ] **Step 6: Update FTS runner initialization to use workload kind**

In `_init_fts_search_runner()`, pass:

```python
workload_kind=WorkloadKind.FULL_TEXT_BM25,
```

to both `SerialSearchRunner` and `MultiProcessingSearchRunner`.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
pytest tests/test_fts_runners.py tests/test_bench_runner.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add vectordb_bench/backend/task_runner.py tests/test_fts_runners.py
git commit -m "refactor: share performance lifecycle for fts"
```

## Task 6: Milvus BM25 Cleanup

**Files:**
- Modify: `vectordb_bench/backend/clients/milvus/config.py`
- Modify: `vectordb_bench/backend/clients/milvus/milvus.py`
- Test: `tests/test_fts_milvus.py`

- [ ] **Step 1: Write failing Milvus unit tests**

Create `tests/test_fts_milvus.py`:

```python
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus


def test_milvus_fts_config_uses_analyzer_max_token_length():
    config = MilvusFtsConfig(
        analyzer_tokenizer="standard",
        analyzer_enable_lowercase=True,
        analyzer_max_token_length=12,
        analyzer_stop_words="the,and",
    )

    params = config.index_param()["analyzer_params"]

    assert params["tokenizer"] == "standard"
    assert "lowercase" in params["filter"]
    assert {"type": "length", "max": 12} in params["filter"]
    assert {"type": "stop", "stop_words": ["the", "and"]} in params["filter"]


def test_milvus_declares_full_text_support():
    assert Milvus.supports_full_text_search() is True
```

- [ ] **Step 2: Run test to verify it fails where behavior is missing**

Run:

```bash
pytest tests/test_fts_milvus.py -q
```

Expected: FAIL until `supports_full_text_search()` is added.

- [ ] **Step 3: Add support declaration**

In `vectordb_bench/backend/clients/milvus/milvus.py`, add to `class Milvus`:

```python
    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True
```

- [ ] **Step 4: Change FTS primary field to string**

In Milvus FTS schema creation, replace:

```python
FieldSchema(name=self._primary_field, dtype=DataType.INT64, is_primary=True),
```

with:

```python
FieldSchema(name=self._primary_field, dtype=DataType.VARCHAR, max_length=512, is_primary=True),
```

Update `insert_documents()` type to:

```python
        doc_ids: list[str],
```

and build rows with:

```python
self._primary_field: str(doc_ids[i])
```

Update `search_documents()` return type to `list[str]` and return:

```python
return [str(hit.entity.get(self._primary_field)) for hit in hits]
```

- [ ] **Step 5: Apply analyzer params to the text field**

When building the `FieldSchema` for text, replace hardcoded:

```python
analyzer_params={"type": "english"},
```

with:

```python
analyzer_params=self.case_config.index_param().get("analyzer_params", {"type": "english"}),
```

- [ ] **Step 6: Run Milvus tests**

Run:

```bash
pytest tests/test_fts_milvus.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vectordb_bench/backend/clients/milvus/config.py vectordb_bench/backend/clients/milvus/milvus.py tests/test_fts_milvus.py
git commit -m "fix: clean up milvus bm25 fts support"
```

## Task 7: ElasticCloud BM25 Adapter

**Files:**
- Modify: `vectordb_bench/backend/clients/elastic_cloud/config.py`
- Modify: `vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py`
- Test: `tests/test_fts_elastic_cloud.py`

- [ ] **Step 1: Write failing ElasticCloud tests**

Create `tests/test_fts_elastic_cloud.py`:

```python
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.elastic_cloud.elastic_cloud import ElasticCloud


def test_elastic_cloud_fts_config_defaults():
    config = ElasticCloudFtsConfig()

    assert config.index_param()["properties"]["text"]["type"] == "text"
    assert config.search_param() == {}


def test_elastic_cloud_declares_full_text_support():
    assert ElasticCloud.supports_full_text_search() is True


def test_elastic_cloud_search_documents_builds_match_query(monkeypatch):
    db = ElasticCloud.__new__(ElasticCloud)
    db.indice = "idx"
    db.id_col_name = "doc_id"
    db.text_col_name = "text"
    calls = {}

    class Client:
        def search(self, **kwargs):
            calls.update(kwargs)
            return {"hits": {"hits": [{"fields": {"doc_id": ["d1"]}}]}}

    db.client = Client()

    assert db.search_documents("hello world", k=3) == ["d1"]
    assert calls["index"] == "idx"
    assert calls["query"] == {"match": {"text": "hello world"}}
    assert calls["size"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_elastic_cloud.py -q
```

Expected: FAIL because `ElasticCloudFtsConfig` and FTS methods do not exist.

- [ ] **Step 3: Add ElasticCloud FTS config**

In `vectordb_bench/backend/clients/elastic_cloud/config.py`, add:

```python
class ElasticCloudFtsConfig(BaseModel, DBCaseConfig):
    number_of_shards: int = 1
    number_of_replicas: int = 0
    refresh_interval: str = "30s"
    use_force_merge: bool = True

    def index_param(self) -> dict:
        return {
            "properties": {
                "doc_id": {"type": "keyword", "store": True},
                "text": {"type": "text"},
            },
        }

    def search_param(self) -> dict:
        return {}
```

- [ ] **Step 4: Update ElasticCloud client for FTS mode**

In `vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py`, import `ElasticCloudFtsConfig` and set in `__init__`:

```python
        self._is_fts = isinstance(db_case_config, ElasticCloudFtsConfig)
        self.text_col_name = "text"
        if self._is_fts:
            self.id_col_name = "doc_id"
```

Add:

```python
    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True
```

In `_create_indice()`, branch:

```python
        if self._is_fts:
            mappings = self.case_config.index_param()
            settings = {
                "index": {
                    "number_of_shards": self.case_config.number_of_shards,
                    "number_of_replicas": self.case_config.number_of_replicas,
                    "refresh_interval": self.case_config.refresh_interval,
                }
            }
            client.indices.create(index=self.indice, mappings=mappings, settings=settings)
            return
```

Add:

```python
    def insert_documents(
        self,
        texts: Iterable[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.client is not None, "should self.init() first"
        docs = list(texts)
        actions = [
            {
                "_index": self.indice,
                "_id": str(doc_ids[i]),
                "_source": {
                    self.id_col_name: str(doc_ids[i]),
                    self.text_col_name: docs[i],
                },
            }
            for i in range(len(docs))
        ]
        try:
            result = bulk(self.client, actions)
            return result[0], None
        except Exception as e:
            log.warning(f"Failed to insert FTS docs: {self.indice} error: {e!s}")
            return 0, e

    def search_documents(
        self,
        query: str,
        k: int = 100,
        **kwargs,
    ) -> list[str]:
        assert self.client is not None, "should self.init() first"
        res = self.client.search(
            index=self.indice,
            query={"match": {self.text_col_name: query}},
            size=k,
            _source=False,
            docvalue_fields=[self.id_col_name],
            stored_fields="_none_",
            filter_path=[f"hits.hits.fields.{self.id_col_name}"],
        )
        return [str(hit["fields"][self.id_col_name][0]) for hit in res["hits"]["hits"]]
```

- [ ] **Step 5: Run ElasticCloud tests**

Run:

```bash
pytest tests/test_fts_elastic_cloud.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/elastic_cloud/config.py vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py tests/test_fts_elastic_cloud.py
git commit -m "feat: add elasticcloud bm25 fts adapter"
```

## Task 8: Vespa BM25 Adapter

**Files:**
- Modify: `vectordb_bench/backend/clients/vespa/config.py`
- Modify: `vectordb_bench/backend/clients/vespa/vespa.py`
- Test: `tests/test_fts_vespa.py`

- [ ] **Step 1: Write failing Vespa tests**

Create `tests/test_fts_vespa.py`:

```python
from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig
from vectordb_bench.backend.clients.vespa.vespa import Vespa


def test_vespa_fts_config_defaults():
    config = VespaFtsConfig()

    assert config.index_param() == {}
    assert config.search_param() == {}


def test_vespa_declares_full_text_support():
    assert Vespa.supports_full_text_search() is True


def test_vespa_search_documents_uses_user_query():
    db = Vespa.__new__(Vespa)
    db.schema_name = "docs"
    calls = {}

    class Result:
        def get_json(self):
            return {"root": {"children": [{"fields": {"id": "d1"}}]}}

    class Client:
        def query(self, query):
            calls.update(query)
            return Result()

    db.client = Client()

    assert db.search_documents("hello world", k=4) == ["d1"]
    assert "userQuery()" in calls["yql"]
    assert calls["query"] == "hello world"
    assert calls["ranking"] == "bm25"
    assert calls["hits"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_vespa.py -q
```

Expected: FAIL because `VespaFtsConfig` and FTS methods do not exist.

- [ ] **Step 3: Add Vespa FTS config**

In `vectordb_bench/backend/clients/vespa/config.py`, add:

```python
class VespaFtsConfig(BaseModel, DBCaseConfig):
    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
```

- [ ] **Step 4: Add FTS mode to Vespa client**

In `vectordb_bench/backend/clients/vespa/vespa.py`, import `VespaFtsConfig` and set:

```python
        self._is_fts = isinstance(self.case_config, VespaFtsConfig)
```

Add:

```python
    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True
```

Add methods:

```python
    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.client is not None
        data = (
            {"id": str(doc_id), "fields": {"id": str(doc_id), "text": text}}
            for doc_id, text in zip(doc_ids, texts, strict=True)
        )
        self.client.feed_iterable(data, self.schema_name)
        return len(texts), None

    def search_documents(
        self,
        query: str,
        k: int = 100,
        **kwargs,
    ) -> list[str]:
        assert self.client is not None
        yql = f"select id from {self.schema_name} where userQuery()"  # noqa: S608
        result = self.client.query(
            {
                "yql": yql,
                "query": query,
                "type": "any",
                "ranking": "bm25",
                "hits": k,
            }
        )
        children = result.get_json().get("root", {}).get("children", [])
        return [str(child["fields"]["id"]) for child in children]
```

In `_create_application_package()`, branch for FTS before vector fields:

```python
        if self._is_fts:
            fields = [
                Field("id", "string", indexing=["summary", "attribute"]),
                Field("text", "string", indexing=["index", "summary"], index="enable-bm25"),
            ]
            return ApplicationPackage(
                "vectordbbench",
                [
                    Schema(
                        self.schema_name,
                        Document(fields),
                        rank_profiles=[RankProfile(name="bm25", first_phase="bm25(text)", inherits="default")],
                    )
                ],
                validations=[Validation(ValidationID.fieldTypeChange, until=tomorrow)],
            )
```

- [ ] **Step 5: Run Vespa tests**

Run:

```bash
pytest tests/test_fts_vespa.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/vespa/config.py vectordb_bench/backend/clients/vespa/vespa.py tests/test_fts_vespa.py
git commit -m "feat: add vespa bm25 fts adapter"
```

## Task 9: Turbopuffer BM25 Adapter

**Files:**
- Modify: `vectordb_bench/backend/clients/turbopuffer/config.py`
- Modify: `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py`
- Test: `tests/test_fts_turbopuffer.py`

- [ ] **Step 1: Write failing Turbopuffer tests**

Create `tests/test_fts_turbopuffer.py`:

```python
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferFtsConfig
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer


def test_turbopuffer_fts_config_defaults():
    config = TurboPufferFtsConfig()

    assert config.index_param() == {}
    assert config.search_param() == {}


def test_turbopuffer_declares_full_text_support():
    assert TurboPuffer.supports_full_text_search() is True


def test_turbopuffer_search_documents_uses_bm25_rank_by():
    db = TurboPuffer.__new__(TurboPuffer)
    calls = {}

    class Row:
        def __init__(self, id):
            self.id = id

    class Result:
        rows = [Row("d1")]

    class Namespace:
        def query(self, **kwargs):
            calls.update(kwargs)
            return Result()

    db.ns = Namespace()

    assert db.search_documents("hello", k=7) == ["d1"]
    assert calls["rank_by"] == ("text", "BM25", "hello")
    assert calls["top_k"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_turbopuffer.py -q
```

Expected: FAIL because `TurboPufferFtsConfig` and FTS methods do not exist.

- [ ] **Step 3: Add Turbopuffer FTS config**

In `vectordb_bench/backend/clients/turbopuffer/config.py`, add:

```python
class TurboPufferFtsConfig(BaseModel, DBCaseConfig):
    time_wait_warmup: int = 60

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
```

- [ ] **Step 4: Add FTS support to Turbopuffer client**

In `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py`, import `TurboPufferFtsConfig`, set:

```python
        self._is_fts = isinstance(db_case_config, TurboPufferFtsConfig)
        self._text_field = "text"
```

Add:

```python
    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True
```

Add:

```python
    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            self.ns.write(
                columns={
                    "id": [str(doc_id) for doc_id in doc_ids],
                    self._text_field: list(texts),
                },
                schema={
                    self._text_field: {
                        "type": "string",
                        "full_text_search": True,
                    }
                },
            )
        except Exception as e:
            log.warning(f"Failed to insert FTS docs. Error: {e}")
            return 0, e
        return len(doc_ids), None

    def search_documents(
        self,
        query: str,
        k: int = 100,
        **kwargs,
    ) -> list[str]:
        res = self.ns.query(
            rank_by=(self._text_field, "BM25", query),
            top_k=k,
        )
        return [str(row.id) for row in res.rows] if res.rows is not None else []
```

- [ ] **Step 5: Run Turbopuffer tests**

Run:

```bash
pytest tests/test_fts_turbopuffer.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/turbopuffer/config.py vectordb_bench/backend/clients/turbopuffer/turbopuffer.py tests/test_fts_turbopuffer.py
git commit -m "feat: add turbopuffer bm25 fts adapter"
```

## Task 10: Config Mapping, Assembly Gating, And Requirements

**Files:**
- Modify: `vectordb_bench/backend/clients/__init__.py`
- Modify: `vectordb_bench/backend/assembler.py`
- Modify: `install/requirements_py3.11.txt`
- Test: `tests/test_fts_backend_capability.py`

- [ ] **Step 1: Extend capability tests**

Append to `tests/test_fts_backend_capability.py`:

```python
import pytest

from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferFtsConfig
from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig
from vectordb_bench.models import CaseConfig, TaskConfig


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
        db_config=DB.QdrantLocal.config_cls(),
        db_case_config=DB.QdrantLocal.case_config_cls()(),
        case_config=CaseConfig(case_id=CaseType.FTSmsmarcoPerformance),
    )

    with pytest.raises(ValueError, match="does not support full-text search"):
        Assembler.assemble("run", task, source=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_backend_capability.py -q
```

Expected: FAIL until config classes are mapped and assembler gates unsupported DBs.

- [ ] **Step 3: Map FTS config classes in DB enum helpers**

In `vectordb_bench/backend/clients/__init__.py`, update `case_config_cls` logic for FTS-capable DBs. For `DB.ElasticCloud`, replace its existing branch with:

```python
        if self == DB.ElasticCloud:
            if index_type == IndexType.FTS_AUTOINDEX:
                from .elastic_cloud.config import ElasticCloudFtsConfig

                return ElasticCloudFtsConfig
            from .elastic_cloud.config import ElasticCloudIndexConfig

            return ElasticCloudIndexConfig
```

For `DB.Vespa`, replace its existing branch with:

```python
        if self == DB.Vespa:
            if index_type == IndexType.FTS_AUTOINDEX:
                from .vespa.config import VespaFtsConfig

                return VespaFtsConfig
            from .vespa.config import VespaHNSWConfig

            return VespaHNSWConfig
```

For `DB.TurboPuffer`, replace its existing branch with:

```python
        if self == DB.TurboPuffer:
            if index_type == IndexType.FTS_AUTOINDEX:
                from .turbopuffer.config import TurboPufferFtsConfig

                return TurboPufferFtsConfig
            from .turbopuffer.config import TurboPufferIndexConfig

            return TurboPufferIndexConfig
```

Keep the current Milvus `_milvus_case_config.get(index_type)` branch unchanged because it already maps `IndexType.FTS_AUTOINDEX` to `MilvusFtsConfig`.

- [ ] **Step 4: Add assembler FTS capability gate**

In `vectordb_bench/backend/assembler.py`, after constructing `c`, add:

```python
        if c.label == CaseLabel.FullTextSearchPerformance and not task.db.init_cls.supports_full_text_search():
            msg = f"{task.db.value} does not support full-text search"
            raise ValueError(msg)
```

- [ ] **Step 5: Add `ir_datasets` runtime requirement**

In `install/requirements_py3.11.txt`, add a separate line:

```text
ir_datasets
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_fts_backend_capability.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vectordb_bench/backend/clients/__init__.py vectordb_bench/backend/assembler.py install/requirements_py3.11.txt tests/test_fts_backend_capability.py
git commit -m "feat: gate fts backend capability"
```

## Task 11: UI And CLI Case Matrix

**Files:**
- Modify: `vectordb_bench/frontend/config/dbCaseConfigs.py`
- Modify: `vectordb_bench/cli/cli.py`
- Test: `tests/test_fts_cases.py`

- [ ] **Step 1: Add UI/CLI case generation tests**

Append to `tests/test_fts_cases.py`:

```python
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.config.dbCaseConfigs import CASE_CONFIG_MAP, get_fts_case_items


def test_fts_case_items_expose_small_medium_by_default_and_large_as_advanced():
    items = get_fts_case_items()
    labels = [item.label for item in items]

    assert "FTS BM25 - MS MARCO Small (100K documents)" in labels
    assert "FTS BM25 - MS MARCO Medium (1M documents)" in labels
    assert "FTS BM25 - HotpotQA Small (100K documents)" in labels
    assert "FTS BM25 - HotpotQA Medium (1M documents)" in labels
    assert "FTS BM25 - MS MARCO Large (8.8M documents)" not in labels
    assert "FTS BM25 - HotpotQA Large (5.2M documents)" not in labels


def test_fts_config_map_contains_supported_backends_only():
    for db in [DB.Milvus, DB.ElasticCloud, DB.Vespa, DB.TurboPuffer]:
        assert CaseLabel.FullTextSearchPerformance in CASE_CONFIG_MAP[db]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_fts_cases.py -q
```

Expected: FAIL because `get_fts_case_items()` does not exist and only Milvus has FTS config.

- [ ] **Step 3: Add FTS case item helper**

In `vectordb_bench/frontend/config/dbCaseConfigs.py`, import `FtsDatasetWithSizeType` and add:

```python
def generate_fts_case(dataset_with_size_type: FtsDatasetWithSizeType) -> list[CaseConfig]:
    return [
        CaseConfig(
            case_id=CaseType.FTSmsmarcoPerformance,
            custom_case={"dataset_with_size_type": dataset_with_size_type.value},
        )
    ]


def get_fts_case_items(include_advanced: bool = False) -> list[UICaseItem]:
    dataset_options = [
        dataset
        for dataset in FtsDatasetWithSizeType
        if include_advanced or not dataset.is_advanced
    ]
    return [
        UICaseItem(
            label=f"FTS BM25 - {dataset.value}",
            description=(
                f"This case tests native BM25 full-text search performance on {dataset.value}. "
                "It reports recall, NDCG, MRR, serial latency, and QPS."
            ),
            cases=generate_fts_case(dataset),
            caseLabel=CaseLabel.FullTextSearchPerformance,
        )
        for dataset in dataset_options
    ]
```

Replace the existing single FTS cluster's `uiCaseItems` argument with:

```python
uiCaseItems=get_fts_case_items(),
```

- [ ] **Step 4: Add FTS config map entries**

In `vectordb_bench/frontend/config/dbCaseConfigs.py`, define empty input lists for first-pass backend FTS configs:

```python
ElasticCloudFtsConfig = []
VespaFtsConfig = []
TurboPufferFtsConfig = []
```

In `CASE_CONFIG_MAP`, add `CaseLabel.FullTextSearchPerformance` to the existing `DB.ElasticCloud`, `DB.Vespa`, and `DB.TurboPuffer` dictionaries:

```python
CaseLabel.FullTextSearchPerformance: ElasticCloudFtsConfig,
CaseLabel.FullTextSearchPerformance: VespaFtsConfig,
CaseLabel.FullTextSearchPerformance: TurboPufferFtsConfig,
```

Use existing `MilvusFtsConfig` for Milvus.

- [ ] **Step 5: Add CLI dataset parameter pass-through**

In `vectordb_bench/cli/cli.py`, update `get_custom_case_config()` so `FTSmsmarcoPerformance` copies `dataset_with_size_type`:

```python
    elif parameters["case_type"] == "FTSmsmarcoPerformance":
        custom_case_config = {
            "dataset_with_size_type": parameters["dataset_with_size_type"],
        }
```

Ensure the click option help includes FTS dataset values:

```python
"MS MARCO Small (100K documents)|MS MARCO Medium (1M documents)|HotpotQA Small (100K documents)|HotpotQA Medium (1M documents)"
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_fts_cases.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vectordb_bench/frontend/config/dbCaseConfigs.py vectordb_bench/cli/cli.py tests/test_fts_cases.py
git commit -m "feat: expose fts bm25 case matrix"
```

## Task 12: End-To-End Verification

**Files:**
- Modify only files required by failures from this task.

- [ ] **Step 1: Run focused FTS test suite**

Run:

```bash
pytest tests/test_fts_dataset.py tests/test_fts_cases.py tests/test_fts_runners.py tests/test_fts_backend_capability.py tests/test_fts_milvus.py tests/test_fts_elastic_cloud.py tests/test_fts_vespa.py tests/test_fts_turbopuffer.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing regression tests that touch changed paths**

Run:

```bash
pytest tests/test_dataset.py tests/test_bench_runner.py tests/test_elasticsearch_cloud.py tests/test_models.py -q
```

Expected: PASS, or existing environment-dependent integration tests are skipped/xfail according to current test behavior. Do not mask new unit failures.

- [ ] **Step 3: Run import checks**

Run:

```bash
python - <<'PY'
import vectordb_bench.backend.dataset
import vectordb_bench.backend.task_runner
import vectordb_bench.frontend.config.dbCaseConfigs
print("imports ok")
PY
```

Expected:

```text
imports ok
```

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files are modified.

- [ ] **Step 5: Handle final verification fixes**

When Steps 1-4 reveal implementation fixes, return to the task that introduced the failing area, add a concrete regression test there, and commit the corrected task files with that task's commit command. Do not create a catch-all verification commit.

If no fixes were needed, leave the branch unchanged after verification.

## Deferred Follow-Ups

- Add FTS `text` payload retrieval after PR 775 lands, reusing shared payload metadata and runner plumbing.
- Add ClickHouse token/boolean FTS cases as a separate benchmark family.
- Add Chinese mMARCO as a multilingual analyzer/tokenizer stress dataset.
- Add dense+BM25 hybrid search using `WorkloadKind.HYBRID_DENSE_BM25`, composable dense/text configs, and a ranker config.

## Self-Review

- Spec coverage: The plan covers dataset correctness, BM25-only scope, ElasticCloud/Vespa/Turbopuffer/Milvus backends, IDs-only first pass, Large as advanced/opt-in, explicit workload kind, optional FTS methods, shared lifecycle, UI/CLI matrix, and tests.
- Placeholder scan: No task relies on unspecified behavior; where backend SDK behavior can vary, the plan gives exact minimal code and tests for the intended call shape.
- Type consistency: FTS document and query IDs are strings end-to-end; `search_documents()` returns `list[str]`; `WorkloadKind.FULL_TEXT_BM25` is used by runners and task orchestration.
