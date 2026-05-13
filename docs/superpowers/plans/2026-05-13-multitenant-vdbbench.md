# Multitenant VDBBench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an end-to-end multitenant benchmark case for Turbopuffer, Pinecone, and Zilliz Cloud.

**Architecture:** Add `MultiTenantPerformanceCase` as a normal performance case with tenant metadata. Extend existing insert and search runners to pass optional tenant context through the existing `VectorDB` client API. Target clients route tenant context to namespaces for Turbopuffer/Pinecone and to partition-key scalar labels for Zilliz Cloud.

**Tech Stack:** Python, Pydantic models, Click CLI, pytest, existing VDBBench runner/client abstractions.

---

## File Structure

- Modify `vectordb_bench/backend/cases.py`: add `CaseType.MultiTenantPerformanceCase`, helper methods for deterministic tenant labels, and case registration.
- Modify `vectordb_bench/backend/clients/api.py`: add optional `tenant_labels_data`, `tenant`, and multitenant readiness hooks to the abstract API.
- Modify `vectordb_bench/backend/runner/concurrent_runner.py`: derive tenant labels from metadata and pass them during insert.
- Modify `vectordb_bench/backend/runner/serial_runner.py`: pass random tenant context to search and skip recall for multitenant cases.
- Modify `vectordb_bench/backend/runner/mp_runner.py`: pass random tenant context during concurrent search.
- Modify `vectordb_bench/backend/task_runner.py`: pass tenant metadata into runners and force `ground_truth=None` for multitenant serial search.
- Modify `vectordb_bench/backend/clients/turbopuffer/config.py` and `turbopuffer.py`: add tenant namespace routing, drop, insert, search, and readiness.
- Modify `vectordb_bench/backend/clients/pinecone/config.py` and `pinecone.py`: add tenant namespace routing, safer drop, insert, search, and readiness.
- Modify `vectordb_bench/backend/clients/milvus/milvus.py`: route `tenant` as a scalar label filter and enforce partition-key use for the multitenant case.
- Modify `vectordb_bench/cli/cli.py`: expose multitenant case parameters.
- Add `tests/test_multitenant_case.py`: case and tenant-label tests.
- Add or extend runner/client tests in `tests/test_cloud_insert_case.py`, `tests/test_cloud_payload_search.py`, `tests/test_milvus.py`, and `tests/test_pinecone_multitenant.py`.

## Task 1: Add Multitenant Case Model

**Files:**
- Modify: `vectordb_bench/backend/cases.py`
- Test: `tests/test_multitenant_case.py`

- [ ] **Step 1: Write failing case tests**

Create `tests/test_multitenant_case.py`:

```python
from vectordb_bench.backend.cases import CaseType, MultiTenantPerformanceCase
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.filter import FilterOp


def test_multitenant_case_defaults_to_cohere_large_1000_tenants():
    case = MultiTenantPerformanceCase()

    assert case.case_id == CaseType.MultiTenantPerformanceCase
    assert case.dataset_with_size_type == DatasetWithSizeType.CohereLarge
    assert case.dataset.data.size == 10_000_000
    assert case.tenant_count == 1000
    assert case.tenant_prefix == "tenant_"
    assert case.tenant_id_width == 4
    assert case.measure_recall is False
    assert case.is_multitenant is True
    assert case.filters.type == FilterOp.NonFilter


def test_multitenant_case_accepts_dataset_and_tenant_count():
    case = MultiTenantPerformanceCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        tenant_count=7,
        tenant_prefix="acct_",
        tenant_id_width=2,
    )

    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.size == 100_000
    assert case.tenant_count == 7
    assert case.tenant_for_id(0) == "acct_00"
    assert case.tenant_for_id(8) == "acct_01"
    assert case.tenant_labels_for_ids([0, 1, 8, 13]) == ["acct_00", "acct_01", "acct_01", "acct_06"]


def test_case_config_constructs_multitenant_case():
    case = CaseType.MultiTenantPerformanceCase.case_cls(
        {
            "dataset_with_size_type": DatasetWithSizeType.CohereSmall.value,
            "tenant_count": 5,
        }
    )

    assert isinstance(case, MultiTenantPerformanceCase)
    assert case.tenant_labels() == ["tenant_0000", "tenant_0001", "tenant_0002", "tenant_0003", "tenant_0004"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_multitenant_case.py -q
```

Expected: FAIL because `MultiTenantPerformanceCase` and `CaseType.MultiTenantPerformanceCase` do not exist.

- [ ] **Step 3: Implement the case**

In `vectordb_bench/backend/cases.py`, add enum value after `CloudInsertCase`:

```python
    MultiTenantPerformanceCase = 700
```

Add the class after `CloudInsertCase` and before `LabelFilterPerformanceCase`:

```python
class MultiTenantPerformanceCase(PerformanceCase):
    case_id: CaseType = CaseType.MultiTenantPerformanceCase
    dataset_with_size_type: DatasetWithSizeType = DatasetWithSizeType.CohereLarge
    tenant_count: int = 1000
    tenant_prefix: str = "tenant_"
    tenant_id_width: int = 4
    tenant_distribution: str = "uniform_by_id_mod"
    measure_recall: bool = False

    def __init__(
        self,
        dataset_with_size_type: DatasetWithSizeType | str = DatasetWithSizeType.CohereLarge,
        tenant_count: int = 1000,
        tenant_prefix: str = "tenant_",
        tenant_id_width: int = 4,
        **kwargs,
    ):
        if not isinstance(dataset_with_size_type, DatasetWithSizeType):
            dataset_with_size_type = DatasetWithSizeType(dataset_with_size_type)
        if tenant_count <= 0:
            raise ValueError("tenant_count must be greater than 0")
        if tenant_id_width <= 0:
            raise ValueError("tenant_id_width must be greater than 0")

        dataset = dataset_with_size_type.get_manager()
        super().__init__(
            name=f"Multi-Tenant - {dataset_with_size_type.value}, {tenant_count} tenants",
            description=(
                "Multi-tenant QPS/latency benchmark with deterministic tenant routing "
                f"({dataset_with_size_type.value}, {tenant_count} tenants)."
            ),
            dataset=dataset,
            load_timeout=dataset_with_size_type.get_load_timeout(),
            optimize_timeout=dataset_with_size_type.get_optimize_timeout(),
            dataset_with_size_type=dataset_with_size_type,
            tenant_count=tenant_count,
            tenant_prefix=tenant_prefix,
            tenant_id_width=tenant_id_width,
            **kwargs,
        )

    @property
    def is_multitenant(self) -> bool:
        return True

    def tenant_for_id(self, row_id: int) -> str:
        tenant_id = int(row_id) % self.tenant_count
        return f"{self.tenant_prefix}{tenant_id:0{self.tenant_id_width}d}"

    def tenant_labels_for_ids(self, row_ids: list[int]) -> list[str]:
        return [self.tenant_for_id(row_id) for row_id in row_ids]

    def tenant_labels(self) -> list[str]:
        return [
            f"{self.tenant_prefix}{tenant_id:0{self.tenant_id_width}d}"
            for tenant_id in range(self.tenant_count)
        ]
```

Add a default property to `Case`:

```python
    @property
    def is_multitenant(self) -> bool:
        return False
```

Register the new case in `type2case`:

```python
    CaseType.MultiTenantPerformanceCase: MultiTenantPerformanceCase,
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_multitenant_case.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/backend/cases.py tests/test_multitenant_case.py
git commit -m "feat: add multitenant performance case"
```

## Task 2: Extend VectorDB API With Optional Tenant Context

**Files:**
- Modify: `vectordb_bench/backend/clients/api.py`
- Test: `tests/test_multitenant_case.py`

- [ ] **Step 1: Add API compatibility tests**

Append to `tests/test_multitenant_case.py`:

```python
from contextlib import contextmanager

from vectordb_bench.backend.clients.api import EmptyDBCaseConfig, VectorDB


class TenantApiProbeDB(VectorDB):
    name = "TenantApiProbeDB"

    def __init__(self, dim=2, db_config=None, db_case_config=None, collection_name="test", drop_old=False, **kwargs):
        self.insert_calls = []
        self.search_calls = []

    @contextmanager
    def init(self):
        yield

    def insert_embeddings(self, embeddings, metadata, labels_data=None, tenant_labels_data=None, **kwargs):
        self.insert_calls.append((embeddings, metadata, labels_data, tenant_labels_data))
        return len(embeddings), None

    def search_embedding(self, query, k=100, payload_profile=None, tenant=None):
        self.search_calls.append((query, k, payload_profile, tenant))
        return []

    def optimize(self, data_size=None):
        return None


def test_vector_db_accepts_optional_tenant_context():
    db = TenantApiProbeDB(db_case_config=EmptyDBCaseConfig())

    count, err = db.insert_embeddings([[0.1, 0.2]], [42], tenant_labels_data=["tenant_0002"])
    result = db.search_embedding([0.1, 0.2], tenant="tenant_0002")

    assert count == 1
    assert err is None
    assert result == []
    assert db.insert_calls[0][3] == ["tenant_0002"]
    assert db.search_calls[0][3] == "tenant_0002"
```

- [ ] **Step 2: Run tests to verify current failure or type mismatch**

Run:

```bash
pytest tests/test_multitenant_case.py::test_vector_db_accepts_optional_tenant_context -q
```

Expected: FAIL if abstract signatures reject the override or static checks complain during collection.

- [ ] **Step 3: Update abstract signatures**

In `vectordb_bench/backend/clients/api.py`, update `insert_embeddings`:

```python
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
```

Update `search_embedding`:

```python
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[int]:
```

Add default multitenant hooks:

```python
    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels

    def supports_multitenant(self) -> bool:
        return False
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_multitenant_case.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/backend/clients/api.py tests/test_multitenant_case.py
git commit -m "feat: add tenant context to vector db API"
```

## Task 3: Derive Tenant Labels During Insert

**Files:**
- Modify: `vectordb_bench/backend/runner/concurrent_runner.py`
- Modify: `vectordb_bench/backend/task_runner.py`
- Test: `tests/test_cloud_insert_case.py`

- [ ] **Step 1: Write failing insert-runner test**

Append to `tests/test_cloud_insert_case.py`:

```python
from contextlib import contextmanager
from unittest.mock import MagicMock

import pandas as pd

from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner


class TenantInsertProbeDB:
    name = "TenantInsertProbeDB"
    thread_safe = True

    def __init__(self):
        self.calls = []

    @contextmanager
    def init(self):
        yield

    def insert_embeddings(self, embeddings, metadata, labels_data=None, tenant_labels_data=None):
        self.calls.append(
            {
                "embeddings": embeddings,
                "metadata": metadata,
                "labels_data": labels_data,
                "tenant_labels_data": tenant_labels_data,
            }
        )
        return len(embeddings), None


class TenantAwareCase:
    is_multitenant = True

    def tenant_labels_for_ids(self, row_ids):
        return [f"tenant_{int(row_id) % 3:04d}" for row_id in row_ids]


def test_concurrent_insert_runner_passes_tenant_labels():
    db = TenantInsertProbeDB()
    dataset = MagicMock()
    dataset.data.train_id_field = "id"
    dataset.data.train_vector_field = "emb"
    dataset.iter_batches.return_value = iter(
        [
            pd.DataFrame(
                {
                    "id": [0, 1, 5],
                    "emb": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                }
            )
        ]
    )

    runner = ConcurrentInsertRunner(
        db=db,
        dataset=dataset,
        normalize=False,
        max_workers=1,
        batch_size=10,
        tenant_case=TenantAwareCase(),
    )

    count = runner.task()

    assert count == 3
    assert db.calls[0]["metadata"] == [0, 1, 5]
    assert db.calls[0]["tenant_labels_data"] == ["tenant_0000", "tenant_0001", "tenant_0002"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_cloud_insert_case.py::test_concurrent_insert_runner_passes_tenant_labels -q
```

Expected: FAIL because `ConcurrentInsertRunner.__init__` does not accept `tenant_case`.

- [ ] **Step 3: Update ConcurrentInsertRunner**

In `vectordb_bench/backend/runner/concurrent_runner.py`, add constructor parameter:

```python
        tenant_case=None,
```

Set it:

```python
        self.tenant_case = tenant_case
```

Update `_insert_batch_with_retry`:

```python
    def _insert_batch_with_retry(
        self,
        db: api.VectorDB,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        retry_idx: int = 0,
    ) -> int:
        insert_count, error = db.insert_embeddings(
            embeddings=embeddings,
            metadata=metadata,
            labels_data=labels_data,
            tenant_labels_data=tenant_labels_data,
        )
```

Update retry call:

```python
                return self._insert_batch_with_retry(
                    db,
                    embeddings,
                    metadata,
                    labels_data,
                    tenant_labels_data,
                    retry_idx,
                )
```

Update `_worker_insert`:

```python
    def _worker_insert(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
    ) -> int:
        db = self._get_thread_db()
        return self._insert_batch_with_retry(db, embeddings, metadata, labels_data, tenant_labels_data)
```

Update `_next_batch` return type and tenant derivation:

```python
    def _next_batch(self) -> tuple[list[list[float]], list[int], list[str] | None, list[str] | None] | None:
```

Add before return:

```python
        tenant_labels_data = None
        if self.tenant_case is not None and getattr(self.tenant_case, "is_multitenant", False):
            tenant_labels_data = self.tenant_case.tenant_labels_for_ids(all_metadata)

        return all_embeddings, all_metadata, labels_data, tenant_labels_data
```

Update `_worker_loop` unpacking:

```python
            embeddings, metadata, labels_data, tenant_labels_data = batch
            total += self._worker_insert(embeddings, metadata, labels_data, tenant_labels_data)
```

- [ ] **Step 4: Pass tenant case from task runner**

In `vectordb_bench/backend/task_runner.py`, update `ConcurrentInsertRunner` construction in `_run_cloud_insert_case` and `_load_train_data`:

```python
                tenant_case=self.ca if self.ca.is_multitenant else None,
```

In `init_db`, pass tenant labels into the DB constructor before `drop_old` cleanup runs. Replace the final `db_cls(...)` call's keyword expansion with:

```python
        extra_db_kwargs = {}
        if collection_name:
            extra_db_kwargs["collection_name"] = collection_name
        if self.ca.is_multitenant:
            extra_db_kwargs["multitenant_tenant_labels"] = self.ca.tenant_labels()

        self.db = db_cls(
            dim=self.ca.dataset.data.dim,
            db_config=db_config_dict,
            db_case_config=self.config.db_case_config,
            drop_old=drop_old,
            with_scalar_labels=self.ca.with_scalar_labels or self.ca.is_multitenant,
            **extra_db_kwargs,
        )
```

In `_pre_run`, after `self.init_db(drop_old)`, keep runtime context synchronized:

```python
            if self.ca.is_multitenant and self.db is not None:
                self.db.set_multitenant_context(self.ca.tenant_labels())
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
pytest tests/test_cloud_insert_case.py::test_concurrent_insert_runner_passes_tenant_labels tests/test_multitenant_case.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/runner/concurrent_runner.py vectordb_bench/backend/task_runner.py tests/test_cloud_insert_case.py
git commit -m "feat: pass tenant labels during insert"
```

## Task 4: Add Tenant-Aware Search Runner Behavior

**Files:**
- Modify: `vectordb_bench/backend/runner/serial_runner.py`
- Modify: `vectordb_bench/backend/runner/mp_runner.py`
- Modify: `vectordb_bench/backend/task_runner.py`
- Test: `tests/test_cloud_payload_search.py`

- [ ] **Step 1: Write failing search runner tests**

Append to `tests/test_cloud_payload_search.py`:

```python
from contextlib import contextmanager

from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner


class TenantSearchProbeDB:
    name = "TenantSearchProbeDB"

    def __init__(self):
        self.tenants = []

    def supports_payload_profile(self, payload_profile):
        return True

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        return None

    def search_embedding(self, query, k=100, payload_profile=None, tenant=None):
        self.tenants.append(tenant)
        return []


def test_serial_search_runner_passes_tenant_and_skips_recall():
    db = TenantSearchProbeDB()
    runner = SerialSearchRunner(
        db=db,
        test_data=[[1.0, 0.0], [0.0, 1.0]],
        ground_truth=None,
        tenant_labels=["tenant_0000", "tenant_0001"],
        measure_recall=False,
    )

    recall, ndcg, p99, p95 = runner.search((runner.test_data, runner.ground_truth))

    assert recall == 0
    assert ndcg == 0
    assert p99 >= 0
    assert p95 >= 0
    assert set(db.tenants).issubset({"tenant_0000", "tenant_0001"})
    assert db.tenants
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_cloud_payload_search.py::test_serial_search_runner_passes_tenant_and_skips_recall -q
```

Expected: FAIL because `SerialSearchRunner` does not accept `tenant_labels` or `measure_recall`.

- [ ] **Step 3: Update SerialSearchRunner**

In `vectordb_bench/backend/runner/serial_runner.py`, add import:

```python
import random
```

Add constructor parameters:

```python
        tenant_labels: list[str] | None = None,
        measure_recall: bool = True,
```

Set fields:

```python
        self.tenant_labels = tenant_labels or []
        self.measure_recall = measure_recall
```

Update `_search_embedding`:

```python
    def _search_embedding(self, emb: list[float], tenant: str | None = None) -> list[int]:
        if tenant is None:
            if self.payload_profile == PayloadProfile.IDS_ONLY:
                return self.db.search_embedding(emb, self.k)
            return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile)
        if self.payload_profile == PayloadProfile.IDS_ONLY:
            return self.db.search_embedding(emb, self.k, tenant=tenant)
        return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile, tenant=tenant)
```

In `search`, before the loop:

```python
            rng = random.Random(0)
```

Inside the loop before timing:

```python
                tenant = self.tenant_labels[rng.randrange(len(self.tenant_labels))] if self.tenant_labels else None
```

Call search:

```python
                    results = self._get_db_search_res(emb, tenant=tenant)
```

Update `_get_db_search_res` signature and retry:

```python
    def _get_db_search_res(self, emb: list[float], tenant: str | None = None, retry_idx: int = 0) -> list[int]:
        try:
            results = self._search_embedding(emb, tenant=tenant)
        except Exception as e:
            log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(emb=emb, tenant=tenant, retry_idx=retry_idx + 1)
```

Change recall block:

```python
                if self.measure_recall and ground_truth is not None:
                    gt = ground_truth[idx]
                    recalls.append(calc_recall(self.k, gt[: self.k], results))
                    ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
                else:
                    recalls.append(0)
                    ndcgs.append(0)
```

- [ ] **Step 4: Update MultiProcessingSearchRunner**

In `vectordb_bench/backend/runner/mp_runner.py`, add constructor parameter:

```python
        tenant_labels: list[str] | None = None,
```

Set:

```python
        self.tenant_labels = tenant_labels or []
```

Update `_search_embedding`:

```python
    def _search_embedding(self, emb: list[float], tenant: str | None = None) -> list[int]:
        if tenant is None:
            if self.payload_profile == PayloadProfile.IDS_ONLY:
                return self.db.search_embedding(emb, self.k)
            return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile)
        if self.payload_profile == PayloadProfile.IDS_ONLY:
            return self.db.search_embedding(emb, self.k, tenant=tenant)
        return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile, tenant=tenant)
```

Inside `search`, after `num, idx = ...`:

```python
            tenant_rng = random.Random(mp.current_process().pid or 0)
```

Before calling `_search_embedding`:

```python
                    tenant = (
                        self.tenant_labels[tenant_rng.randrange(len(self.tenant_labels))]
                        if self.tenant_labels
                        else None
                    )
                    self._search_embedding(test_data[idx], tenant=tenant)
```

- [ ] **Step 5: Pass tenant labels from task runner**

In `vectordb_bench/backend/task_runner.py`, compute in `_init_search_runner`:

```python
        tenant_labels = self.ca.tenant_labels() if self.ca.is_multitenant else None
        measure_recall = getattr(self.ca, "measure_recall", True)
        gt_df = self.ca.dataset.gt_data if measure_recall else None
```

Pass into `SerialSearchRunner`:

```python
                tenant_labels=tenant_labels,
                measure_recall=measure_recall,
```

Pass into `MultiProcessingSearchRunner`:

```python
                tenant_labels=tenant_labels,
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_cloud_payload_search.py::test_serial_search_runner_passes_tenant_and_skips_recall tests/test_multitenant_case.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vectordb_bench/backend/runner/serial_runner.py vectordb_bench/backend/runner/mp_runner.py vectordb_bench/backend/task_runner.py tests/test_cloud_payload_search.py
git commit -m "feat: pass tenant context during search"
```

## Task 5: Implement Zilliz/Milvus Partition-Key Routing

**Files:**
- Modify: `vectordb_bench/backend/clients/milvus/milvus.py`
- Test: `tests/test_milvus.py`

- [ ] **Step 1: Write failing Milvus tenant filter test**

Append to `tests/test_milvus.py`:

```python
from types import SimpleNamespace


def test_milvus_multitenant_search_uses_tenant_label_filter():
    from vectordb_bench.backend.clients.milvus.milvus import Milvus
    from vectordb_bench.backend.payload import PayloadProfile

    db = object.__new__(Milvus)
    db.client = SimpleNamespace(search=lambda **kwargs: [[{"pk": 1}]])
    db.collection_name = "test_collection"
    db._vector_field = "vector"
    db._primary_field = "pk"
    db._scalar_label_field = "label"
    db.case_config = SimpleNamespace(search_param=lambda: {"metric_type": "COSINE"})
    db.expr = ""

    result = db.search_embedding([0.1, 0.2], k=3, payload_profile=PayloadProfile.IDS_ONLY, tenant="tenant_0003")

    assert result == [1]
```

- [ ] **Step 2: Run test to verify current signature failure**

Run:

```bash
pytest tests/test_milvus.py::test_milvus_multitenant_search_uses_tenant_label_filter -q
```

Expected: FAIL because `Milvus.search_embedding` does not accept `tenant`.

- [ ] **Step 3: Update Milvus search signature and filter composition**

In `vectordb_bench/backend/clients/milvus/milvus.py`, update signature:

```python
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[int]:
```

Before `search_kwargs`, add:

```python
        expr = self.expr
        if tenant is not None:
            tenant_expr = f"{self._scalar_label_field} == '{tenant}'"
            expr = tenant_expr if not expr else f"({expr}) and ({tenant_expr})"
```

Use:

```python
            "filter": expr,
```

- [ ] **Step 4: Enforce partition key for multitenant Zilliz/Milvus**

In `vectordb_bench/backend/task_runner.py`, after `self.init_db(drop_old)` and before dataset prepare:

```python
            if self.ca.is_multitenant and self.config.db in {DB.Milvus, DB.ZillizCloud}:
                if not getattr(self.config.db_case_config, "use_partition_key", False):
                    raise ValueError("MultiTenantPerformanceCase requires use_partition_key=True for Milvus/ZillizCloud")
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_milvus.py::test_milvus_multitenant_search_uses_tenant_label_filter -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/milvus/milvus.py vectordb_bench/backend/task_runner.py tests/test_milvus.py
git commit -m "feat: route multitenant zilliz searches by partition key"
```

## Task 6: Implement Turbopuffer Namespace Routing

**Files:**
- Modify: `vectordb_bench/backend/clients/turbopuffer/config.py`
- Modify: `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py`
- Test: `tests/test_multitenant_case.py`

- [ ] **Step 1: Add Turbopuffer namespace grouping tests**

Append to `tests/test_multitenant_case.py`:

```python
from types import SimpleNamespace


class FakeTurboNamespace:
    def __init__(self):
        self.write_calls = []
        self.query_calls = []

    def write(self, **kwargs):
        self.write_calls.append(kwargs)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(rows=[SimpleNamespace(id="10")])

    def metadata(self):
        return {"index": {"unindexed_bytes": 0}}


class FakeTurboClient:
    def __init__(self):
        self.namespaces = {}

    def namespace(self, name):
        self.namespaces.setdefault(name, FakeTurboNamespace())
        return self.namespaces[name]


def test_turbopuffer_groups_multitenant_insert_and_search(monkeypatch):
    from vectordb_bench.backend.clients.api import MetricType
    from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
    from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer

    fake_client = FakeTurboClient()
    monkeypatch.setattr(TurboPuffer, "_create_client", lambda self: fake_client)

    db = TurboPuffer(
        dim=2,
        db_config={
            "api_key": "k",
            "region": "r",
            "api_base_url": None,
            "namespace": "single",
            "multitenant_namespace_prefix": "mt_",
        },
        db_case_config=TurboPufferIndexConfig(metric_type=MetricType.COSINE),
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
    assert result == [10]
    assert fake_client.namespaces["mt_tenant_0000"].write_calls[0]["upsert_columns"]["id"] == [0, 2]
    assert fake_client.namespaces["mt_tenant_0001"].write_calls[0]["upsert_columns"]["id"] == [1]
    assert fake_client.namespaces["mt_tenant_0001"].query_calls
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_multitenant_case.py::test_turbopuffer_groups_multitenant_insert_and_search -q
```

Expected: FAIL because config and client do not support `multitenant_namespace_prefix` or tenant routing.

- [ ] **Step 3: Add config field**

In `vectordb_bench/backend/clients/turbopuffer/config.py`, add:

```python
    multitenant_namespace_prefix: str = "vdbbench_mt_"
```

Include in `to_dict`:

```python
            "multitenant_namespace_prefix": self.multitenant_namespace_prefix,
```

- [ ] **Step 4: Implement namespace routing**

In `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py`, set in `__init__`:

```python
        self.multitenant_namespace_prefix = db_config.get("multitenant_namespace_prefix", "vdbbench_mt_")
        self.multitenant_tenant_labels: list[str] = kwargs.get("multitenant_tenant_labels", [])
        self._ns_cache = {}
```

Add helpers:

```python
    def supports_multitenant(self) -> bool:
        return True

    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels

    def _namespace_name_for_tenant(self, tenant: str | None) -> str:
        if tenant is None:
            return self.namespace
        return f"{self.multitenant_namespace_prefix}{tenant}"

    def _namespace_for_tenant(self, tenant: str | None):
        name = self._namespace_name_for_tenant(tenant)
        ns = self._ns_cache.get(name)
        if ns is None:
            ns = self.client.namespace(name)
            self._ns_cache[name] = ns
        return ns
```

In `init`, reset cache:

```python
        self._ns_cache = {}
```

In `insert_embeddings`, branch when `tenant_labels_data` is present:

```python
            if tenant_labels_data is not None:
                inserted = 0
                for tenant in sorted(set(tenant_labels_data)):
                    idxs = [i for i, t in enumerate(tenant_labels_data) if t == tenant]
                    tenant_vectors = [vectors[i] for i in idxs]
                    tenant_metadata = [metadata[i] for i in idxs]
                    self._namespace_for_tenant(tenant).write(
                        upsert_columns={
                            self._scalar_id_field: tenant_metadata,
                            self._vector_field: tenant_vectors,
                        },
                        distance_metric=self.metric,
                        disable_backpressure=self.db_case_config.disable_backpressure,
                    )
                    inserted += len(idxs)
                return inserted, None
```

In `search_embedding`, use:

```python
        ns = self._namespace_for_tenant(tenant)
        res = ns.query(**query_kwargs)
```

Update signature to include `tenant`.

In `poll_insert_readiness`, if multitenant labels are set, aggregate `unindexed_bytes` across labels and return `fully_indexed=True` only when all are zero.

In `__init__`, replace single-namespace drop behavior with a branch:

```python
        if drop_old:
            tmp_client = self._create_client()
            if self.multitenant_tenant_labels:
                for tenant in self.multitenant_tenant_labels:
                    try:
                        tmp_client.namespace(self._namespace_name_for_tenant(tenant)).delete_all()
                    except Exception as e:
                        log.warning(f"Failed to delete multitenant namespace {tenant}. Error: {e}")
            else:
                log.info(f"Drop old. delete the namespace: {self.namespace}")
                ns = tmp_client.namespace(self.namespace)
                try:
                    ns.delete_all()
                except Exception as e:
                    log.warning(f"Failed to delete all. Error: {e}")
            tmp_client = None
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_multitenant_case.py::test_turbopuffer_groups_multitenant_insert_and_search -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/turbopuffer/config.py vectordb_bench/backend/clients/turbopuffer/turbopuffer.py tests/test_multitenant_case.py
git commit -m "feat: route turbopuffer multitenant namespaces"
```

## Task 7: Implement Pinecone Namespace Routing

**Files:**
- Modify: `vectordb_bench/backend/clients/pinecone/config.py`
- Modify: `vectordb_bench/backend/clients/pinecone/pinecone.py`
- Test: `tests/test_pinecone_multitenant.py`

- [ ] **Step 1: Write Pinecone namespace tests**

Create `tests/test_pinecone_multitenant.py`:

```python
from types import SimpleNamespace


class FakePineconeIndex:
    def __init__(self):
        self.upserts = []
        self.queries = []

    def describe_index_stats(self):
        return {"dimension": 2, "total_vector_count": 3, "namespaces": {"mt_tenant_0000": {}, "mt_tenant_0001": {}}}

    def upsert(self, vectors, namespace=None):
        self.upserts.append((vectors, namespace))
        return SimpleNamespace(_response_info={"raw_headers": {"x-pinecone-request-lsn": "7"}})

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return {
            "matches": [{"id": "11"}],
            "_response_info": {"raw_headers": {"x-pinecone-max-indexed-lsn": "7"}},
        }

    def delete(self, delete_all=False, namespace=None):
        return None


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
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_pinecone_multitenant.py -q
```

Expected: FAIL because Pinecone config/client do not support tenant namespace routing.

- [ ] **Step 3: Add Pinecone config field**

In `vectordb_bench/backend/clients/pinecone/config.py`, add:

```python
    multitenant_namespace_prefix: str = "vdbbench_mt_"
```

Include in `to_dict`:

```python
            "multitenant_namespace_prefix": self.multitenant_namespace_prefix,
```

- [ ] **Step 4: Implement Pinecone tenant routing**

In `vectordb_bench/backend/clients/pinecone/pinecone.py`, set in `__init__`:

```python
        self.multitenant_namespace_prefix = db_config.get("multitenant_namespace_prefix", "vdbbench_mt_")
        self.multitenant_tenant_labels: list[str] = kwargs.get("multitenant_tenant_labels", [])
        self._drop_old = drop_old
```

Add helpers:

```python
    def supports_multitenant(self) -> bool:
        return True

    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels

    def _namespace_for_tenant(self, tenant: str | None) -> str | None:
        if tenant is None:
            return None
        return f"{self.multitenant_namespace_prefix}{tenant}"
```

In drop-old, delete configured benchmark tenant namespaces when `self.multitenant_tenant_labels` is set. Keep current delete-all-namespaces behavior only for non-multitenant runs:

```python
        if drop_old and self.multitenant_tenant_labels:
            for tenant in self.multitenant_tenant_labels:
                namespace = self._namespace_for_tenant(tenant)
                log.info(f"Pinecone index delete multitenant namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)
        elif drop_old:
            index_stats = index.describe_index_stats()
            index_dim = index_stats["dimension"]
            if index_dim != dim:
                msg = f"Pinecone index {self.index_name} dimension mismatch, expected {index_dim} got {dim}"
                raise ValueError(msg)
            for namespace in index_stats["namespaces"]:
                log.info(f"Pinecone index delete namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)
```

Keep `set_multitenant_context` simple:

```python
    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels
```

In `insert_embeddings`, branch when `tenant_labels_data` is present and call `self.index.upsert(insert_datas, namespace=self._namespace_for_tenant(tenant))`.

In `search_embedding`, add `tenant` to the signature and pass:

```python
            namespace=self._namespace_for_tenant(tenant),
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_pinecone_multitenant.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/pinecone/config.py vectordb_bench/backend/clients/pinecone/pinecone.py tests/test_pinecone_multitenant.py
git commit -m "feat: route pinecone multitenant namespaces"
```

## Task 8: Add CLI Support

**Files:**
- Modify: `vectordb_bench/cli/cli.py`
- Test: `tests/test_cloud_insert_case.py`

- [ ] **Step 1: Add CLI config test**

Append to `tests/test_cloud_insert_case.py`:

```python
def test_cli_builds_multitenant_custom_case_config():
    from vectordb_bench.cli.cli import get_custom_case_config

    cfg = get_custom_case_config(
        {
            "case_type": "MultiTenantPerformanceCase",
            "dataset_with_size_type": "Small Cohere (768dim, 100K)",
            "tenant_count": 13,
            "tenant_prefix": "acct_",
            "tenant_id_width": 3,
        }
    )

    assert cfg == {
        "dataset_with_size_type": "Small Cohere (768dim, 100K)",
        "tenant_count": 13,
        "tenant_prefix": "acct_",
        "tenant_id_width": 3,
    }
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_cloud_insert_case.py::test_cli_builds_multitenant_custom_case_config -q
```

Expected: FAIL because `get_custom_case_config` does not handle `MultiTenantPerformanceCase`.

- [ ] **Step 3: Update CLI parameter mapping**

In `vectordb_bench/cli/cli.py`, add branch in `get_custom_case_config`:

```python
    elif parameters["case_type"] == "MultiTenantPerformanceCase":
        custom_case_config = {
            "dataset_with_size_type": parameters["dataset_with_size_type"],
            "tenant_count": parameters["tenant_count"],
            "tenant_prefix": parameters["tenant_prefix"],
            "tenant_id_width": parameters["tenant_id_width"],
        }
```

Add options to `CommonTypedDict`:

```python
    tenant_count: Annotated[
        int,
        click.option(
            "--tenant-count",
            type=int,
            default=1000,
            show_default=True,
            help="Tenant count for MultiTenantPerformanceCase",
        ),
    ]
    tenant_prefix: Annotated[
        str,
        click.option(
            "--tenant-prefix",
            type=str,
            default="tenant_",
            show_default=True,
            help="Tenant label prefix for MultiTenantPerformanceCase",
        ),
    ]
    tenant_id_width: Annotated[
        int,
        click.option(
            "--tenant-id-width",
            type=int,
            default=4,
            show_default=True,
            help="Zero-padding width for MultiTenantPerformanceCase tenant IDs",
        ),
    ]
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_cloud_insert_case.py::test_cli_builds_multitenant_custom_case_config tests/test_multitenant_case.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/cli/cli.py tests/test_cloud_insert_case.py
git commit -m "feat: expose multitenant case cli options"
```

## Task 9: Integration Verification

**Files:**
- Modify only if tests reveal a defect in prior tasks.

- [ ] **Step 1: Run focused multitenant tests**

Run:

```bash
pytest tests/test_multitenant_case.py tests/test_pinecone_multitenant.py tests/test_milvus.py::test_milvus_multitenant_search_uses_tenant_label_filter -q
```

Expected: PASS.

- [ ] **Step 2: Run runner and CLI regression tests**

Run:

```bash
pytest tests/test_cloud_insert_case.py tests/test_cloud_payload_search.py -q
```

Expected: PASS.

- [ ] **Step 3: Run a broader backend-safe test subset**

Run:

```bash
pytest tests/test_dataset.py tests/test_data_source.py tests/test_db_client_resolution.py tests/test_models.py -q
```

Expected: PASS.

- [ ] **Step 4: Inspect diff for accidental result-file changes**

Run:

```bash
git status --short
git diff --stat
```

Expected: only source/test files from the multitenant implementation are staged or modified by this work. Existing unrelated dirty files may still appear; do not add them.

- [ ] **Step 5: Final commit if previous tasks left uncommitted fixes**

```bash
git add vectordb_bench tests
git commit -m "test: verify multitenant benchmark integration"
```

Skip this commit when there are no uncommitted implementation fixes after Task 8.

## Self-Review Notes

Spec coverage:

- End-to-end load/readiness/search is covered by Tasks 3, 4, 6, 7, and 9.
- Turbopuffer namespace routing is covered by Task 6.
- Pinecone namespace routing is covered by Task 7.
- Zilliz partition-key routing is covered by Task 5.
- Generic dataset and tenant count are covered by Task 1 and Task 8.
- QPS/latency-only behavior is covered by Task 4.
- CLI support is covered by Task 8.

The plan intentionally does not add frontend UI, recall ground-truth generation, zipfian tenants, noisy-neighbor tests, or per-tenant parquet pre-splitting because these are out of scope for v1.
