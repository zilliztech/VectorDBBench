# Cloud Cold Latency Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CloudColdLatencyCase`, which runs 1000 serial cold searches followed immediately by the same 1000 warm searches and records first-query, p99, p95, average latency, and cold/warm ratios.

**Architecture:** Add a dedicated case label and case class so cold latency does not overload normal performance metrics. Add a focused `ColdWarmSearchRunner` for two-pass latency-only search, then route it through `CaseRunner` and CLI custom-case parsing. Store the exact result JSON in `Metric.additional_parameters["cold_latency"]` while preserving existing payload metadata fields.

**Tech Stack:** Python, Pydantic-style project `BaseModel`, pytest, NumPy, Click/Typer annotations, existing VDBBench runner and case patterns.

---

## File Structure

- Create `vectordb_bench/backend/runner/cold_warm_runner.py`: implements `ColdWarmSearchRunner` and owns cold/warm latency stats.
- Modify `vectordb_bench/backend/runner/__init__.py`: exports `ColdWarmSearchRunner`.
- Modify `vectordb_bench/backend/cases.py`: adds `CaseType.CloudColdLatencyCase`, `CaseLabel.CloudColdLatency`, and `CloudColdLatencyCase`.
- Modify `vectordb_bench/backend/task_runner.py`: imports the new runner, routes `CaseLabel.CloudColdLatency`, and stores result JSON in `Metric.additional_parameters`.
- Modify `vectordb_bench/backend/assembler.py`: keeps cloud cold runners in the assembled task list and checks filter support.
- Modify `vectordb_bench/cli/cli.py`: parses `--cloud-cold-query-count` and builds the custom case config.
- Create `tests/test_cloud_cold_latency_case.py`: covers case construction, runner behavior, task runner integration, and CLI parsing.

## Task 1: Case Model And CLI Config

**Files:**
- Modify: `vectordb_bench/backend/cases.py`
- Modify: `vectordb_bench/cli/cli.py`
- Test: `tests/test_cloud_cold_latency_case.py`

- [ ] **Step 1: Write failing case and CLI tests**

Create `tests/test_cloud_cold_latency_case.py` with these tests:

```python
import pytest

from vectordb_bench.backend.cases import CaseLabel, CaseType, CloudColdLatencyCase
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.models import CaseConfig


def test_cloud_cold_latency_case_defaults_to_laion_100m():
    case = CloudColdLatencyCase()

    assert case.case_id == CaseType.CloudColdLatencyCase
    assert case.label == CaseLabel.CloudColdLatency
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000
    assert case.dataset.data.dim == 768
    assert case.payload_profile == PayloadProfile.IDS_ONLY
    assert case.query_count == 1000
    assert case.filters.type == FilterOp.NonFilter


def test_cloud_cold_latency_case_accepts_payload_dataset_and_int_filter():
    case = CloudColdLatencyCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        payload_profile="vector",
        filter_rate=0.9,
        query_count=10,
    )

    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.name == "Cohere"
    assert case.payload_profile == PayloadProfile.VECTOR
    assert case.filter_rate == 0.9
    assert case.query_count == 10
    assert case.filters.type == FilterOp.NumGE


def test_cloud_cold_latency_case_accepts_label_filter():
    case = CloudColdLatencyCase(label_percentage=0.9)

    assert case.label_percentage == 0.9
    assert case.filters.type == FilterOp.StrEqual


def test_cloud_cold_latency_case_rejects_two_filter_types():
    with pytest.raises(ValueError, match="supports only one filter type"):
        CloudColdLatencyCase(filter_rate=0.9, label_percentage=0.9)


def test_cloud_cold_latency_case_rejects_invalid_query_count():
    with pytest.raises(ValueError, match="query_count must be positive"):
        CloudColdLatencyCase(query_count=0)


def test_case_config_builds_cloud_cold_latency_case_from_custom_case():
    case = CaseConfig(
        case_id=CaseType.CloudColdLatencyCase,
        custom_case={
            "payload_profile": "scalar_label",
            "label_percentage": 0.9,
            "query_count": 12,
        },
    ).case

    assert isinstance(case, CloudColdLatencyCase)
    assert case.payload_profile == PayloadProfile.SCALAR_LABEL
    assert case.label_percentage == 0.9
    assert case.query_count == 12


def test_cli_builds_cloud_cold_latency_custom_case_config():
    params = {
        "case_type": "CloudColdLatencyCase",
        "payload_profile": "vector",
        "cloud_filter_rate": 0.9,
        "cloud_label_percentage": None,
        "cloud_cold_query_count": 1000,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }

    assert get_custom_case_config(params) == {
        "payload_profile": "vector",
        "filter_rate": 0.9,
        "query_count": 1000,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py -q
```

Expected: FAIL with an import error or attribute error because `CloudColdLatencyCase` and `CaseType.CloudColdLatencyCase` do not exist.

- [ ] **Step 3: Implement the case model**

In `vectordb_bench/backend/cases.py`, add `CloudColdLatencyCase = 700` after `CloudInsertCase = 600` in `CaseType`:

```python
    CloudPayloadSearchCase = 500
    CloudInsertCase = 600
    CloudColdLatencyCase = 700
```

Add `CloudColdLatency = auto()` after `CloudInsert = auto()` in `CaseLabel`:

```python
    CloudInsert = auto()
    CloudColdLatency = auto()
```

Add this class after `CloudPayloadSearchCase` and before `CloudInsertCase`:

```python
class CloudColdLatencyCase(Case):
    case_id: CaseType = CaseType.CloudColdLatencyCase
    label: CaseLabel = CaseLabel.CloudColdLatency
    dataset_with_size_type: DatasetWithSizeType | None = None
    payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY
    filter_rate: float | None = None
    label_percentage: float | None = None
    query_count: int = 1000

    def __init__(
        self,
        dataset_with_size_type: DatasetWithSizeType | str | None = None,
        payload_profile: PayloadProfile | str = PayloadProfile.IDS_ONLY,
        filter_rate: float | None = None,
        label_percentage: float | None = None,
        query_count: int = 1000,
        **kwargs,
    ):
        if filter_rate is not None and label_percentage is not None:
            msg = "CloudColdLatencyCase supports only one filter type per run"
            raise ValueError(msg)
        if query_count <= 0:
            msg = "query_count must be positive"
            raise ValueError(msg)
        if dataset_with_size_type is not None and not isinstance(dataset_with_size_type, DatasetWithSizeType):
            dataset_with_size_type = DatasetWithSizeType(dataset_with_size_type)
        if not isinstance(payload_profile, PayloadProfile):
            payload_profile = PayloadProfile(payload_profile)

        if dataset_with_size_type is None:
            dataset = Dataset.LAION.manager(100_000_000)
            load_timeout = config.LOAD_TIMEOUT_768D_100M
            optimize_timeout = config.OPTIMIZE_TIMEOUT_768D_100M
            dataset_name = "LAION 100M (768dim)"
        else:
            dataset = dataset_with_size_type.get_manager()
            load_timeout = dataset_with_size_type.get_load_timeout()
            optimize_timeout = dataset_with_size_type.get_optimize_timeout()
            dataset_name = dataset_with_size_type.value

        name = f"Cloud Cold Latency - {payload_profile.value} - {dataset_name}"
        description = (
            "Cloud leaderboard cold/warm serial latency case with explicit response payload profile. "
            f"Payload profile: {payload_profile.value}; dataset: {dataset_name}; queries: {query_count}."
        )
        super().__init__(
            name=name,
            description=description,
            dataset=dataset,
            load_timeout=load_timeout,
            optimize_timeout=optimize_timeout,
            dataset_with_size_type=dataset_with_size_type,
            payload_profile=payload_profile,
            filter_rate=filter_rate,
            label_percentage=label_percentage,
            query_count=query_count,
            **kwargs,
        )

    @property
    def filters(self) -> Filter:
        if self.label_percentage is not None:
            return LabelFilter(label_percentage=self.label_percentage)
        if self.filter_rate is None:
            return non_filter
        int_field = self.dataset.data.train_id_field
        int_value = int(self.dataset.data.size * self.filter_rate)
        return NewIntFilter(filter_rate=self.filter_rate, int_field=int_field, int_value=int_value)
```

Add it to `type2case`:

```python
    CaseType.CloudPayloadSearchCase: CloudPayloadSearchCase,
    CaseType.CloudInsertCase: CloudInsertCase,
    CaseType.CloudColdLatencyCase: CloudColdLatencyCase,
```

- [ ] **Step 4: Implement CLI custom-case parsing**

In `vectordb_bench/cli/cli.py`, update `get_custom_case_config()` after the `CloudInsertCase` branch:

```python
    elif parameters["case_type"] == "CloudColdLatencyCase":
        custom_case_config = {
            "payload_profile": parameters["payload_profile"],
            "query_count": parameters["cloud_cold_query_count"],
            "dataset_with_size_type": parameters["dataset_with_size_type"],
        }
        if parameters["cloud_filter_rate"] is not None:
            custom_case_config["filter_rate"] = parameters["cloud_filter_rate"]
        if parameters["cloud_label_percentage"] is not None:
            custom_case_config["label_percentage"] = parameters["cloud_label_percentage"]
```

Add this field to `CommonTypedDict` after `cloud_insert_duration`:

```python
    cloud_cold_query_count: Annotated[
        int,
        click.option(
            "--cloud-cold-query-count",
            type=int,
            default=1000,
            show_default=True,
            help="Number of serial queries per cold/warm pass for CloudColdLatencyCase",
        ),
    ]
```

- [ ] **Step 5: Run case and CLI tests**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py -q
```

Expected: PASS for the seven tests in this task.

- [ ] **Step 6: Commit**

Run:

```bash
git add vectordb_bench/backend/cases.py vectordb_bench/cli/cli.py tests/test_cloud_cold_latency_case.py
git commit -m "Add cloud cold latency case model"
```

## Task 2: ColdWarmSearchRunner

**Files:**
- Create: `vectordb_bench/backend/runner/cold_warm_runner.py`
- Modify: `vectordb_bench/backend/runner/__init__.py`
- Test: `tests/test_cloud_cold_latency_case.py`

- [ ] **Step 1: Add failing runner tests**

Append these tests to `tests/test_cloud_cold_latency_case.py`:

```python
from contextlib import contextmanager

import numpy as np

from vectordb_bench.backend.runner.cold_warm_runner import ColdWarmSearchRunner


class FakeColdWarmDB:
    name = "FakeColdWarmDB"

    def __init__(self, supported_payload_profiles=None):
        self.supported_payload_profiles = supported_payload_profiles or {PayloadProfile.IDS_ONLY}
        self.calls = []
        self.prepare_filter_calls = []
        self.init_enter_count = 0

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile in self.supported_payload_profiles

    @contextmanager
    def init(self):
        self.init_enter_count += 1
        yield

    def prepare_filter(self, filters):
        self.prepare_filter_calls.append(filters)

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        self.calls.append((query, k, kwargs))
        return list(range(k))


def test_cold_warm_runner_computes_stats_and_ratios(monkeypatch):
    db = FakeColdWarmDB()
    # Cold latencies: 0.2, 0.2, 0.2. Warm latencies: 0.1, 0.1, 0.1.
    perf_values = iter([0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9])
    monkeypatch.setattr("vectordb_bench.backend.runner.cold_warm_runner.time.perf_counter", lambda: next(perf_values))

    runner = ColdWarmSearchRunner(
        db=db,
        test_data=[[0.1], [0.2], [0.3]],
        k=3,
        query_count=3,
    )

    result = runner.run()

    assert result == {
        "cold_stats": {
            "first_query_latency": 0.2,
            "p99_latency": 0.2,
            "p95_latency": 0.2,
            "avg_latency": 0.2,
        },
        "warm_stats": {
            "first_query_latency": 0.1,
            "p99_latency": 0.1,
            "p95_latency": 0.1,
            "avg_latency": 0.1,
        },
        "cold_warm_ratio": {
            "first_query_latency_ratio": 2.0,
            "p99_latency_ratio": 2.0,
            "p95_latency_ratio": 2.0,
            "avg_latency_ratio": 2.0,
        },
    }
    assert db.init_enter_count == 1
    assert len(db.prepare_filter_calls) == 1
    assert [call[0] for call in db.calls] == [[0.1], [0.2], [0.3], [0.1], [0.2], [0.3]]


def test_cold_warm_runner_passes_payload_profile_in_both_passes(monkeypatch):
    db = FakeColdWarmDB(supported_payload_profiles={PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR})
    perf_values = iter([0.0, 0.1, 0.1, 0.2])
    monkeypatch.setattr("vectordb_bench.backend.runner.cold_warm_runner.time.perf_counter", lambda: next(perf_values))

    runner = ColdWarmSearchRunner(
        db=db,
        test_data=[np.array([0.1])],
        k=3,
        payload_profile=PayloadProfile.VECTOR,
        query_count=1,
    )

    runner.run()

    assert db.calls == [
        ([0.1], 3, {"payload_profile": PayloadProfile.VECTOR}),
        ([0.1], 3, {"payload_profile": PayloadProfile.VECTOR}),
    ]


def test_cold_warm_runner_fails_for_unsupported_payload_profile():
    db = FakeColdWarmDB()

    with pytest.raises(NotImplementedError, match="payload_profile=vector"):
        ColdWarmSearchRunner(
            db=db,
            test_data=[[0.1]],
            payload_profile=PayloadProfile.VECTOR,
            query_count=1,
        )


def test_cold_warm_runner_rejects_too_few_queries():
    db = FakeColdWarmDB()

    with pytest.raises(ValueError, match="query_count=2 exceeds test_data size=1"):
        ColdWarmSearchRunner(db=db, test_data=[[0.1]], query_count=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py::test_cold_warm_runner_computes_stats_and_ratios -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vectordb_bench.backend.runner.cold_warm_runner'`.

- [ ] **Step 3: Implement the runner**

Create `vectordb_bench/backend/runner/cold_warm_runner.py`:

```python
import logging
import time

import numpy as np

from vectordb_bench import config
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.filter import Filter, non_filter
from vectordb_bench.backend.payload import PayloadProfile

log = logging.getLogger(__name__)


class ColdWarmSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        k: int = 100,
        filters: Filter = non_filter,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        query_count: int = 1000,
    ):
        if query_count <= 0:
            msg = "query_count must be positive"
            raise ValueError(msg)
        if len(test_data) < query_count:
            msg = f"query_count={query_count} exceeds test_data size={len(test_data)}"
            raise ValueError(msg)

        self.db = db
        self.k = k
        self.filters = filters
        self.payload_profile = payload_profile
        self.query_count = query_count

        if not self.db.supports_payload_profile(self.payload_profile):
            msg = f"{self.db.name} does not support payload_profile={self.payload_profile.value}"
            raise NotImplementedError(msg)

        selected = test_data[:query_count]
        if selected and isinstance(selected[0], np.ndarray):
            self.test_data = [query.tolist() for query in selected]
        else:
            self.test_data = selected

    def _search_embedding(self, emb: list[float]) -> list[int]:
        if self.payload_profile == PayloadProfile.IDS_ONLY:
            return self.db.search_embedding(emb, self.k)
        return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile)

    def _get_db_search_res(self, emb: list[float], retry_idx: int = 0) -> list[int]:
        try:
            return self._search_embedding(emb)
        except Exception as e:
            log.warning(f"Cold/warm search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1)
            msg = f"Cold/warm search failed and retried more than {config.MAX_SEARCH_RETRY} times"
            raise RuntimeError(msg) from e

    @staticmethod
    def _latency_stats(latencies: list[float]) -> dict[str, float]:
        return {
            "first_query_latency": round(latencies[0], 4),
            "p99_latency": round(float(np.percentile(latencies, 99)), 4),
            "p95_latency": round(float(np.percentile(latencies, 95)), 4),
            "avg_latency": round(float(np.mean(latencies)), 4),
        }

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

    @classmethod
    def _ratio_stats(cls, cold_stats: dict[str, float], warm_stats: dict[str, float]) -> dict[str, float]:
        return {
            "first_query_latency_ratio": cls._safe_ratio(
                cold_stats["first_query_latency"], warm_stats["first_query_latency"]
            ),
            "p99_latency_ratio": cls._safe_ratio(cold_stats["p99_latency"], warm_stats["p99_latency"]),
            "p95_latency_ratio": cls._safe_ratio(cold_stats["p95_latency"], warm_stats["p95_latency"]),
            "avg_latency_ratio": cls._safe_ratio(cold_stats["avg_latency"], warm_stats["avg_latency"]),
        }

    def _run_pass(self, pass_name: str) -> dict[str, float]:
        latencies = []
        for idx, emb in enumerate(self.test_data):
            started = time.perf_counter()
            self._get_db_search_res(emb)
            latencies.append(time.perf_counter() - started)
            if len(latencies) % 100 == 0:
                log.debug("%s search_count=%s latest_latency=%s", pass_name, len(latencies), latencies[-1])
        stats = self._latency_stats(latencies)
        log.info("%s latency stats: %s", pass_name, stats)
        return stats

    def run(self) -> dict[str, dict[str, float]]:
        with self.db.init():
            self.db.prepare_filter(self.filters)
            cold_stats = self._run_pass("cold")
            warm_stats = self._run_pass("warm")

        return {
            "cold_stats": cold_stats,
            "warm_stats": warm_stats,
            "cold_warm_ratio": self._ratio_stats(cold_stats, warm_stats),
        }
```

- [ ] **Step 4: Export the runner**

Modify `vectordb_bench/backend/runner/__init__.py`:

```python
from .cold_warm_runner import ColdWarmSearchRunner
from .concurrent_runner import ConcurrentInsertRunner
from .mp_runner import MultiProcessingSearchRunner
from .read_write_runner import ReadWriteRunner
from .serial_runner import SerialInsertRunner, SerialSearchRunner

__all__ = [
    "ColdWarmSearchRunner",
    "ConcurrentInsertRunner",
    "MultiProcessingSearchRunner",
    "ReadWriteRunner",
    "SerialInsertRunner",
    "SerialSearchRunner",
]
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py -q
```

Expected: PASS for all case and runner tests currently in the file.

- [ ] **Step 6: Commit**

Run:

```bash
git add vectordb_bench/backend/runner/cold_warm_runner.py vectordb_bench/backend/runner/__init__.py tests/test_cloud_cold_latency_case.py
git commit -m "Add cold warm search runner"
```

## Task 3: Task Runner And Assembler Integration

**Files:**
- Modify: `vectordb_bench/backend/task_runner.py`
- Modify: `vectordb_bench/backend/assembler.py`
- Test: `tests/test_cloud_cold_latency_case.py`

- [ ] **Step 1: Add failing integration tests**

Append these tests to `tests/test_cloud_cold_latency_case.py`:

```python
from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.models import TaskConfig, TaskStage


def test_assembler_schedules_cloud_cold_latency_case():
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.CloudColdLatencyCase,
            custom_case={"query_count": 1},
        ),
        stages=[TaskStage.SEARCH_SERIAL],
    )

    runner = Assembler.assemble_all("run-id", "task-label", [task], DatasetSource.S3)

    assert len(runner.case_runners) == 1
    assert runner.case_runners[0].ca.label == CaseLabel.CloudColdLatency


def test_case_runner_stores_cloud_cold_latency_metric(monkeypatch):
    case = CloudColdLatencyCase(query_count=1)
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.CloudColdLatencyCase,
            custom_case={"query_count": 1},
        ),
        stages=[TaskStage.SEARCH_SERIAL],
    )
    runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    runner.db = FakeColdWarmDB()
    runner.test_emb = [[0.1]]

    expected = {
        "cold_stats": {
            "first_query_latency": 0.2,
            "p99_latency": 0.2,
            "p95_latency": 0.2,
            "avg_latency": 0.2,
        },
        "warm_stats": {
            "first_query_latency": 0.1,
            "p99_latency": 0.1,
            "p95_latency": 0.1,
            "avg_latency": 0.1,
        },
        "cold_warm_ratio": {
            "first_query_latency_ratio": 2.0,
            "p99_latency_ratio": 2.0,
            "p95_latency_ratio": 2.0,
            "avg_latency_ratio": 2.0,
        },
    }

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            return expected

    monkeypatch.setattr("vectordb_bench.backend.task_runner.ColdWarmSearchRunner", FakeRunner)
    monkeypatch.setattr(
        CaseRunner,
        "_init_cold_warm_search_runner",
        lambda self: setattr(self, "cold_warm_search_runner", FakeRunner()),
    )

    metric = runner._run_cloud_cold_latency_case()

    assert metric.additional_parameters["cold_latency"] == expected
    assert metric.payload_profile == "ids_only"
    assert metric.payload_estimated_bytes_per_query == case.estimated_payload_bytes_per_query(task.case_config.k)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py::test_assembler_schedules_cloud_cold_latency_case tests/test_cloud_cold_latency_case.py::test_case_runner_stores_cloud_cold_latency_metric -q
```

Expected: FAIL because the assembler does not include `CaseLabel.CloudColdLatency` and `CaseRunner` has no `_run_cloud_cold_latency_case()`.

- [ ] **Step 3: Import and add runner state**

In `vectordb_bench/backend/task_runner.py`, update the runner import:

```python
from .runner import (
    ColdWarmSearchRunner,
    ConcurrentInsertRunner,
    MultiProcessingSearchRunner,
    ReadWriteRunner,
    SerialInsertRunner,
    SerialSearchRunner,
)
```

Add this field to `CaseRunner`:

```python
    cold_warm_search_runner: ColdWarmSearchRunner | None = None
```

- [ ] **Step 4: Route the case label**

In `CaseRunner.run()`, add this branch after `CloudInsert`:

```python
        if self.ca.label == CaseLabel.CloudColdLatency:
            return self._run_cloud_cold_latency_case(drop_old)
```

- [ ] **Step 5: Add the cold/warm initialization and run methods**

In `vectordb_bench/backend/task_runner.py`, add these methods near `_run_cloud_insert_case()`:

```python
    def _init_cold_warm_search_runner(self) -> None:
        if self.normalize:
            test_emb = np.stack(self.ca.dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            self.test_emb = test_emb.tolist()
        else:
            self.test_emb = self.ca.dataset.test_data

        self.cold_warm_search_runner = ColdWarmSearchRunner(
            db=self.db,
            test_data=self.test_emb,
            filters=self.ca.filters,
            k=self.config.case_config.k,
            payload_profile=self.ca.payload_profile,
            query_count=self.ca.query_count,
        )

    def _run_cloud_cold_latency_case(self, drop_old: bool = True) -> Metric:
        log.info("Start cloud cold latency case")
        try:
            m = Metric()
            if drop_old:
                if TaskStage.LOAD in self.config.stages:
                    _, load_dur = self._load_train_data()
                    build_dur = self._optimize()
                    m.insert_duration = round(load_dur, 4)
                    m.optimize_duration = round(build_dur, 4)
                    m.load_duration = round(load_dur + build_dur, 4)
                else:
                    log.info("Data loading skipped")

            self._init_cold_warm_search_runner()
            m.additional_parameters = {
                "cold_latency": self.cold_warm_search_runner.run(),
            }
            m.payload_profile = self.ca.payload_profile.value
            m.payload_estimated_bytes_per_query = self.ca.estimated_payload_bytes_per_query(self.config.case_config.k)
        except Exception as e:
            log.warning(f"Failed to run cloud cold latency case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Cloud cold latency case got result: {m}")
            return m
```

- [ ] **Step 6: Update assembler grouping**

In `vectordb_bench/backend/assembler.py`, replace the runner grouping block with:

```python
        load_runners = [r for r in runners if r.ca.label == CaseLabel.Load]
        perf_runners = [r for r in runners if r.ca.label == CaseLabel.Performance]
        streaming_runners = [r for r in runners if r.ca.label == CaseLabel.Streaming]
        cloud_insert_runners = [r for r in runners if r.ca.label == CaseLabel.CloudInsert]
        cloud_cold_latency_runners = [r for r in runners if r.ca.label == CaseLabel.CloudColdLatency]

        search_filter_runners = [*perf_runners, *cloud_cold_latency_runners]
```

Then update the group-by loop to use `search_filter_runners`:

```python
        for r in search_filter_runners:
            db = r.config.db
            if db not in db2runner:
                db2runner[db] = []
            db2runner[db].append(r)
```

Use this exact final assembly block so cloud cold latency runners are appended through `db2runner` only once:

```python
        all_runners = []
        all_runners.extend(load_runners)
        all_runners.extend(streaming_runners)
        all_runners.extend(cloud_insert_runners)
        for v in db2runner.values():
            all_runners.extend(v)
```

This keeps filter-supported search cases grouped by DB and avoids adding the same runner twice.

- [ ] **Step 7: Run integration tests**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py -q
```

Expected: PASS for all tests in the file.

- [ ] **Step 8: Commit**

Run:

```bash
git add vectordb_bench/backend/task_runner.py vectordb_bench/backend/assembler.py tests/test_cloud_cold_latency_case.py
git commit -m "Wire cloud cold latency runner into tasks"
```

## Task 4: Regression And Focused Verification

**Files:**
- Test only: existing changed files from prior tasks

- [ ] **Step 1: Run focused cloud tests**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py tests/test_cloud_payload_case.py tests/test_cloud_insert_case.py -q
```

Expected: PASS.

- [ ] **Step 2: Run runner and assembler tests**

Run:

```bash
pytest tests/test_cloud_cold_latency_case.py tests/test_cloud_payload_search.py -q
```

Expected: PASS.

- [ ] **Step 3: Run static import check**

Run:

```bash
python - <<'PY'
from vectordb_bench.backend.cases import CaseType, CloudColdLatencyCase
from vectordb_bench.backend.runner import ColdWarmSearchRunner
from vectordb_bench.models import CaseConfig

case = CaseConfig(case_id=CaseType.CloudColdLatencyCase).case
assert isinstance(case, CloudColdLatencyCase)
assert ColdWarmSearchRunner.__name__ == "ColdWarmSearchRunner"
print(case.name)
PY
```

Expected output includes:

```text
Cloud Cold Latency - ids_only - LAION 100M (768dim)
```

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat HEAD
git diff -- vectordb_bench/backend/cases.py vectordb_bench/backend/runner/cold_warm_runner.py vectordb_bench/backend/task_runner.py vectordb_bench/backend/assembler.py vectordb_bench/cli/cli.py tests/test_cloud_cold_latency_case.py
```

Expected: only the cloud cold latency files and tests changed.

- [ ] **Step 5: Commit verification-only fixes if needed**

If a verification command exposed a small defect and you fixed it, commit the fix:

```bash
git add vectordb_bench/backend/cases.py vectordb_bench/backend/runner/cold_warm_runner.py vectordb_bench/backend/runner/__init__.py vectordb_bench/backend/task_runner.py vectordb_bench/backend/assembler.py vectordb_bench/cli/cli.py tests/test_cloud_cold_latency_case.py
git commit -m "Fix cloud cold latency verification issues"
```

If no files changed after verification, do not create an empty commit.

## Self-Review

Spec coverage:

- New case type and label: Task 1.
- Payload profile, int filter, label filter, and query count support: Task 1.
- Dedicated cold/warm runner with one DB context and two immediate passes: Task 2.
- First-query, p99, p95, avg, and ratio result JSON: Task 2.
- Store result under `Metric.additional_parameters["cold_latency"]`: Task 3.
- Preserve payload metadata fields: Task 3.
- CLI custom-case config: Task 1.
- Filter support and task assembly: Task 3.
- Regression verification: Task 4.

Placeholder scan:

- The plan contains concrete code-changing steps.
- Each verification step includes commands and expected results.

Type consistency:

- Case type is consistently `CaseType.CloudColdLatencyCase`.
- Case class is consistently `CloudColdLatencyCase`.
- Case label is consistently `CaseLabel.CloudColdLatency`.
- Runner class is consistently `ColdWarmSearchRunner`.
- Result key is consistently `additional_parameters["cold_latency"]`.
