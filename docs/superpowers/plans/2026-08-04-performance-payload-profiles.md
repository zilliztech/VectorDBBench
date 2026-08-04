# Performance Payload Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make IDs-only and vector response payloads first-class, independently measurable options for every vector `PerformanceCase`, with Milvus and Zilliz Cloud frontend support and payload-aware results.

**Architecture:** Add an optional top-level payload field to `CaseConfig` and resolve it into existing case constructor arguments while preserving legacy `custom_case` data. Reuse the existing serial, concurrent, Milvus, and Zilliz Cloud payload paths; add early backend validation, frontend task expansion, and payload-aware result/export identity without creating a new case type.

**Tech Stack:** Python 3.11, Pydantic 2, Click, Streamlit, pytest, Ruff, Black.

---

## File Map

- `vectordb_bench/models.py`: public `CaseConfig` payload contract, validation, legacy resolution, hashing/serialization behavior.
- `vectordb_bench/cli/cli.py`: map the existing CLI option into top-level `CaseConfig` for `PerformanceCase` workloads.
- `vectordb_bench/frontend/config/dbCaseConfigs.py`: identify vector performance items, track selected profiles, and expand one base case into one or two tasks.
- `vectordb_bench/frontend/components/run_test/caseSelector.py`: render the Milvus/Zilliz Cloud return-scenario multiselect.
- `vectordb_bench/backend/task_runner.py`: reject unsupported profiles before dataset preparation or loading.
- `vectordb_bench/frontend/components/check_results/data.py`: include payload in frontend result identity.
- `vectordb_bench/restful/format_res.py`: retain payload fields in REST output.
- `vectordb_bench/results/getLeaderboardDataV2.py`: retain payload identity in the legacy export.
- `README.md`: document shared payload usage next to the large-topK case.
- Existing focused test modules: lock down configuration, CLI, frontend, runtime, Milvus translation, result grouping, serialization, and compatibility.

**Line-budget justification:** The repository already exceeds 1,000 lines. This plan creates no new source module or duplicate benchmark case; it adds the minimum shared configuration, UI, validation, and result wiring needed for the approved cross-cutting contract. Offsetting those additions would require the separately deferred `CloudPayloadSearchCase` refactor, so unrelated removal is intentionally excluded from this implementation.

## Environment Setup

- [ ] **Step 1: Create an isolated Python 3.11 environment**

Run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

Expected: installation completes and `.venv/bin/python -c "import vectordb_bench, pytest, pydantic"` exits with status 0.

- [ ] **Step 2: Confirm the baseline focused tests pass**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_models.py \
  tests/test_cloud_payload_case.py \
  tests/test_large_topk_cli.py \
  tests/test_large_topk_frontend.py \
  tests/test_milvus.py -q
```

Expected: all selected baseline tests pass before source changes.

### Task 1: First-Class CaseConfig Payload Contract

**Files:**
- Modify: `tests/test_models.py`
- Modify: `tests/test_cloud_payload_case.py`
- Modify: `vectordb_bench/models.py:1-248`

- [ ] **Step 1: Write failing CaseConfig tests**

Add imports:

```python
from pydantic import ValidationError

from vectordb_bench.backend.payload import PayloadProfile
```

Add these tests to `tests/test_models.py`:

```python
def test_performance_case_config_applies_top_level_payload_without_mutating_custom_case():
    custom_case = {}
    case_config = CaseConfig(
        case_id=CaseType.Performance768D100M,
        custom_case=custom_case,
        payload_profile=PayloadProfile.VECTOR,
    )

    assert case_config.case.payload_profile == PayloadProfile.VECTOR
    assert custom_case == {}


def test_performance_case_config_payload_round_trip_and_hash_identity():
    ids_only = CaseConfig(
        case_id=CaseType.Performance768D100M,
        payload_profile=PayloadProfile.IDS_ONLY,
    )
    vector = CaseConfig(
        case_id=CaseType.Performance768D100M,
        payload_profile=PayloadProfile.VECTOR,
    )

    restored = CaseConfig.model_validate(vector.model_dump(mode="json"))

    assert restored.payload_profile == PayloadProfile.VECTOR
    assert restored.case.payload_profile == PayloadProfile.VECTOR
    assert hash(ids_only) != hash(vector)


def test_case_config_rejects_payload_for_non_performance_case():
    with pytest.raises(ValidationError, match="only supported for PerformanceCase"):
        CaseConfig(
            case_id=CaseType.CapacityDim128,
            payload_profile=PayloadProfile.VECTOR,
        )
```

Add these compatibility tests to `tests/test_cloud_payload_case.py`:

```python
def test_case_config_preserves_legacy_payload_profile():
    case_config = CaseConfig(
        case_id=CaseType.CloudPayloadSearchCase,
        custom_case={"payload_profile": "vector"},
    )

    assert case_config.payload_profile is None
    assert case_config.case.payload_profile == PayloadProfile.VECTOR


def test_case_config_accepts_matching_top_level_and_legacy_payload_profiles():
    case_config = CaseConfig(
        case_id=CaseType.CloudPayloadSearchCase,
        custom_case={"payload_profile": "vector"},
        payload_profile=PayloadProfile.VECTOR,
    )

    assert case_config.case.payload_profile == PayloadProfile.VECTOR


def test_case_config_rejects_conflicting_payload_profiles():
    with pytest.raises(ValidationError, match="conflicts with custom_case"):
        CaseConfig(
            case_id=CaseType.CloudPayloadSearchCase,
            custom_case={"payload_profile": "ids_only"},
            payload_profile=PayloadProfile.VECTOR,
        )
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_models.py::test_performance_case_config_applies_top_level_payload_without_mutating_custom_case \
  tests/test_models.py::test_performance_case_config_payload_round_trip_and_hash_identity \
  tests/test_models.py::test_case_config_rejects_payload_for_non_performance_case \
  tests/test_cloud_payload_case.py::test_case_config_preserves_legacy_payload_profile \
  tests/test_cloud_payload_case.py::test_case_config_accepts_matching_top_level_and_legacy_payload_profiles \
  tests/test_cloud_payload_case.py::test_case_config_rejects_conflicting_payload_profiles -q
```

Expected: failures report that `CaseConfig` does not accept or apply `payload_profile`.

- [ ] **Step 3: Implement the minimal CaseConfig contract**

Update imports in `vectordb_bench/models.py`:

```python
from pydantic import field_validator, model_validator

from .backend.cases import Case, CaseType, PerformanceCase
from .backend.payload import PayloadProfile
```

Add the field and validator to `CaseConfig`:

```python
class CaseConfig(BaseModel):
    case_id: CaseType
    custom_case: dict | None = None
    payload_profile: PayloadProfile | None = None
    k: int | None = config.K_DEFAULT
    concurrency_search_config: ConcurrencySearchConfig = ConcurrencySearchConfig()

    @model_validator(mode="after")
    def validate_payload_profile(self) -> Self:
        if self.payload_profile is None:
            return self

        case_cls = type2case[self.case_id]
        if not issubclass(case_cls, PerformanceCase):
            msg = "Top-level payload_profile is only supported for PerformanceCase cases"
            raise ValueError(msg)

        legacy_profile = (self.custom_case or {}).get("payload_profile")
        if legacy_profile is not None and PayloadProfile(legacy_profile) != self.payload_profile:
            msg = "Top-level payload_profile conflicts with custom_case payload_profile"
            raise ValueError(msg)
        return self
```

Replace the `case` property with a non-mutating merge:

```python
    @property
    def case(self) -> Case:
        custom_case = dict(self.custom_case or {})
        if self.payload_profile is not None:
            custom_case["payload_profile"] = self.payload_profile
        return self.case_id.case_cls(custom_case or None)
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the command from Step 2.

Expected: `6 passed`.

- [ ] **Step 5: Run related model and payload tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_models.py tests/test_cloud_payload_case.py tests/test_case_runner_reuse.py -q
```

Expected: all tests pass; legacy cloud payload and load-reuse tests remain unchanged.

- [ ] **Step 6: Commit the CaseConfig contract**

```bash
git add vectordb_bench/models.py tests/test_models.py tests/test_cloud_payload_case.py
git diff --cached --check
git -c user.name=jamesgao-jpg -c user.email=james.gao@zilliz.com commit -s -m "feat: add performance payload configuration"
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py --repo . --commit HEAD
```

Expected: commit succeeds and DCO verification prints the exact required sign-off.

### Task 2: CLI Propagation

**Files:**
- Modify: `tests/test_large_topk_cli.py`
- Modify: `vectordb_bench/cli/cli.py:20-34,599-607,876-890`

- [ ] **Step 1: Write a failing CLI propagation test**

Add imports and a capture helper to `tests/test_large_topk_cli.py`:

```python
from pytest import MonkeyPatch

from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.cli import cli as common_cli


def invoke_test_command(monkeypatch: MonkeyPatch, args: list[str]):
    captured = {}

    def fake_run(tasks, task_label):
        captured["task"] = tasks[0]
        captured["task_label"] = task_label

    monkeypatch.setattr(common_cli.benchmark_runner, "run", fake_run)
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    result = CliRunner().invoke(test_cli.Test, args)
    return result, captured
```

Add tests:

```python
def test_cli_applies_vector_payload_to_standard_performance_case(monkeypatch: MonkeyPatch):
    result, captured = invoke_test_command(
        monkeypatch,
        [
            "--case-type",
            "Performance768D100M",
            "--payload-profile",
            "vector",
        ],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["task"].case_config
    assert case_config.payload_profile == PayloadProfile.VECTOR
    assert case_config.case.payload_profile == PayloadProfile.VECTOR


def test_cli_does_not_set_top_level_payload_for_capacity_case(monkeypatch: MonkeyPatch):
    result, captured = invoke_test_command(
        monkeypatch,
        ["--case-type", "CapacityDim128"],
    )

    assert result.exit_code == 0, result.output
    assert captured["task"].case_config.payload_profile is None
```

- [ ] **Step 2: Run the CLI tests and verify the vector test fails**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_large_topk_cli.py::test_cli_applies_vector_payload_to_standard_performance_case \
  tests/test_large_topk_cli.py::test_cli_does_not_set_top_level_payload_for_capacity_case -q
```

Expected: the performance assertion fails because the CLI-created `CaseConfig` has no top-level profile.

- [ ] **Step 3: Add a narrow CLI resolver and pass its value to CaseConfig**

Update imports in `vectordb_bench/cli/cli.py`:

```python
from ..backend.cases import PerformanceCase, type2case
```

Add this helper near `get_custom_case_config`:

```python
def get_case_payload_profile(parameters: dict[str, Any]) -> PayloadProfile | None:
    case_type = CaseType[parameters["case_type"]]
    if not issubclass(type2case[case_type], PerformanceCase):
        return None
    return PayloadProfile(parameters["payload_profile"])
```

Pass it when constructing `CaseConfig`:

```python
        case_config=CaseConfig(
            case_id=CaseType[parameters["case_type"]],
            payload_profile=get_case_payload_profile(parameters),
            k=parameters["k"],
            concurrency_search_config=ConcurrencySearchConfig(
```

Update the option help text:

```python
help="Response payload profile for vector performance, cloud payload, and FTS cases",
```

- [ ] **Step 4: Run CLI compatibility tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_large_topk_cli.py \
  tests/test_cloud_payload_case.py \
  tests/test_cloud_cold_latency_case.py \
  tests/test_multitenant_case.py \
  tests/test_turbopuffer_cli.py \
  tests/test_milvus_zilliz_cli.py -q
```

Expected: all tests pass, including specialized `custom_case` mappings.

- [ ] **Step 5: Commit CLI propagation**

```bash
git add vectordb_bench/cli/cli.py tests/test_large_topk_cli.py
git diff --cached --check
git -c user.name=jamesgao-jpg -c user.email=james.gao@zilliz.com commit -s -m "feat: expose payload profiles in performance CLI"
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py --repo . --commit HEAD
```

Expected: commit and DCO check pass.

### Task 3: Frontend Return-Scenario Expansion

**Files:**
- Modify: `tests/test_large_topk_frontend.py`
- Modify: `vectordb_bench/frontend/config/dbCaseConfigs.py:1-112`
- Modify: `vectordb_bench/frontend/components/run_test/caseSelector.py:1-93`

- [ ] **Step 1: Write failing frontend model tests**

Add imports to `tests/test_large_topk_frontend.py`:

```python
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.frontend.components.run_test.caseSelector import payloadProfileSetting
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UICaseItem,
    generate_normal_cases,
    get_payload_profile_options,
)
```

Add tests:

```python
def test_performance_ui_case_expands_selected_payload_profiles():
    item = UICaseItem(cases=generate_normal_cases(CaseType.Performance768D100M))
    item.payload_profiles = [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]

    cases = item.get_cases()

    assert [case.payload_profile for case in cases] == [
        PayloadProfile.IDS_ONLY,
        PayloadProfile.VECTOR,
    ]
    assert all(case.case_id == CaseType.Performance768D100M for case in cases)


def test_capacity_ui_case_does_not_expand_payload_profiles():
    item = UICaseItem(cases=generate_normal_cases(CaseType.CapacityDim128))
    item.payload_profiles = [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]

    cases = item.get_cases()

    assert len(cases) == 1
    assert cases[0].payload_profile is None


def test_payload_profile_options_require_only_supported_backends():
    assert get_payload_profile_options([DB.Milvus]) == [
        PayloadProfile.IDS_ONLY,
        PayloadProfile.VECTOR,
    ]
    assert get_payload_profile_options([DB.Milvus, DB.ZillizCloud]) == [
        PayloadProfile.IDS_ONLY,
        PayloadProfile.VECTOR,
    ]
    assert get_payload_profile_options([DB.Milvus, DB.Test]) == [PayloadProfile.IDS_ONLY]
    assert get_payload_profile_options([]) == [PayloadProfile.IDS_ONLY]


def test_payload_profile_setting_records_frontend_selection():
    class FakeContainer:
        def __init__(self):
            self.options = []

        def multiselect(self, label, options, default, format_func, key):
            assert label == "Return scenario"
            assert default == [PayloadProfile.IDS_ONLY]
            assert format_func(PayloadProfile.VECTOR) == "Vector payload"
            assert key
            self.options = options
            return options

        def error(self, message):
            raise AssertionError(message)

    item = UICaseItem(cases=generate_normal_cases(CaseType.Performance768D100M))
    container = FakeContainer()

    payloadProfileSetting(container, item, [DB.Milvus])

    assert container.options == [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]
    assert item.payload_profiles == [PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR]
```

- [ ] **Step 2: Run the frontend tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_large_topk_frontend.py::test_performance_ui_case_expands_selected_payload_profiles \
  tests/test_large_topk_frontend.py::test_capacity_ui_case_does_not_expand_payload_profiles \
  tests/test_large_topk_frontend.py::test_payload_profile_options_require_only_supported_backends \
  tests/test_large_topk_frontend.py::test_payload_profile_setting_records_frontend_selection -q
```

Expected: import or attribute failures for the new frontend payload helpers.

- [ ] **Step 3: Add payload state and task expansion to UICaseItem**

Update imports in `dbCaseConfigs.py`:

```python
from pydantic import BaseModel, Field

from vectordb_bench.backend.cases import CaseLabel, CaseType, PerformanceCase
from vectordb_bench.backend.payload import PayloadProfile
```

Add the support constant and option function:

```python
VECTOR_PAYLOAD_SUPPORTED_DBS = {DB.Milvus, DB.ZillizCloud}


def get_payload_profile_options(active_dbs: list[DB]) -> list[PayloadProfile]:
    profiles = [PayloadProfile.IDS_ONLY]
    if active_dbs and all(db in VECTOR_PAYLOAD_SUPPORTED_DBS for db in active_dbs):
        profiles.append(PayloadProfile.VECTOR)
    return profiles
```

Add state and a capability property to `UICaseItem`:

```python
    payload_profiles: list[PayloadProfile] = Field(
        default_factory=lambda: [PayloadProfile.IDS_ONLY],
    )

    @property
    def supports_payload_profiles(self) -> bool:
        return bool(self.cases) and all(isinstance(case.case, PerformanceCase) for case in self.cases)
```

Refactor `get_cases()` so customization happens first and payload expansion happens second:

```python
    def get_cases(self) -> list[CaseConfig]:
        cases = self.cases
        if self.extra_custom_case_config_inputs:
            cases = [
                CaseConfig(
                    case_id=case.case_id,
                    k=case.k,
                    concurrency_search_config=case.concurrency_search_config,
                    custom_case={**case.custom_case, **self.tmp_custom_config},
                )
                for case in cases
            ]
        if not self.supports_payload_profiles:
            return cases
        return [
            case.model_copy(update={"payload_profile": payload_profile})
            for case in cases
            for payload_profile in self.payload_profiles
        ]
```

- [ ] **Step 4: Render the multiselect in caseSelector**

Import the option helper and payload type:

```python
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.frontend.config.dbCaseConfigs import get_payload_profile_options
```

Add the renderer:

```python
PAYLOAD_PROFILE_LABELS = {
    PayloadProfile.IDS_ONLY: "IDs only",
    PayloadProfile.VECTOR: "Vector payload",
}


def payloadProfileSetting(container, uiCaseItem: UICaseItem, active_dbs: list[DB]) -> None:
    if not uiCaseItem.supports_payload_profiles:
        return
    options = get_payload_profile_options(active_dbs)
    selected = [profile for profile in uiCaseItem.payload_profiles if profile in options]
    if not selected:
        selected = [PayloadProfile.IDS_ONLY]
    backend_key = "-".join(sorted(db.name for db in active_dbs)) or "none"
    uiCaseItem.payload_profiles = container.multiselect(
        "Return scenario",
        options=options,
        default=selected,
        format_func=PAYLOAD_PROFILE_LABELS.__getitem__,
        key=f"payload-profile-{uiCaseItem.label}-{backend_key}",
    )
    if not uiCaseItem.payload_profiles:
        container.error("Select at least one return scenario.")
```

Call it only for selected cases:

```python
    if selected:
        payloadProfileSetting(st.container(), uiCaseItem, activedDbList)
        dbCaseConfigSetting(st.container(), dbToCaseClusterConfigs, uiCaseItem, activedDbList)
```

- [ ] **Step 5: Run frontend and task-generation tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_large_topk_frontend.py \
  tests/test_models.py -q
```

Expected: all selected tests pass and profile expansion produces distinct hashable `CaseConfig` values.

- [ ] **Step 6: Commit frontend expansion**

```bash
git add \
  vectordb_bench/frontend/config/dbCaseConfigs.py \
  vectordb_bench/frontend/components/run_test/caseSelector.py \
  tests/test_large_topk_frontend.py
git diff --cached --check
git -c user.name=jamesgao-jpg -c user.email=james.gao@zilliz.com commit -s -m "feat: add performance payload scenarios to frontend"
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py --repo . --commit HEAD
```

Expected: commit and DCO check pass.

### Task 4: Early Runtime Validation and Milvus Contract

**Files:**
- Modify: `tests/test_cloud_payload_case.py`
- Modify: `tests/test_milvus.py`
- Modify: `vectordb_bench/backend/task_runner.py:183-275`

- [ ] **Step 1: Write a failing pre-load validation test**

Add this test to `tests/test_cloud_payload_case.py`:

```python
def test_case_runner_rejects_unsupported_payload_before_dataset_prepare(monkeypatch: pytest.MonkeyPatch):
    events = []
    case_config = CaseConfig(
        case_id=CaseType.Performance768D100M,
        payload_profile=PayloadProfile.VECTOR,
    )
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=case_config,
    )
    runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=case_config.case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )

    monkeypatch.setattr(
        type(runner.ca.dataset),
        "resolve_search_files",
        lambda self, **kwargs: events.append("resolve"),
    )
    monkeypatch.setattr(
        type(runner.ca.dataset),
        "prepare",
        lambda self, *args, **kwargs: events.append("prepare"),
    )

    def fake_init_db(self, drop_old=True):
        events.append("init_db")
        self.db = FakeDB()

    monkeypatch.setattr(CaseRunner, "init_db", fake_init_db)

    with pytest.raises(NotImplementedError, match="payload_profile=vector"):
        runner._pre_run(drop_old=False)

    assert events == ["resolve", "init_db"]
```

- [ ] **Step 2: Add a Milvus vector request translation test**

Add to `tests/test_milvus.py`:

```python
def test_milvus_vector_payload_requests_vector_field_and_returns_ids():
    captured = {}

    def search(**kwargs):
        captured.update(kwargs)
        return [[{"pk": 1, "vector": [0.1, 0.2]}]]

    db = object.__new__(Milvus)
    db.client = SimpleNamespace(search=search)
    db.collection_name = "test_collection"
    db._vector_field = "vector"
    db._primary_field = "pk"
    db._scalar_label_field = "label"
    db.case_config = SimpleNamespace(search_param=lambda: {"metric_type": "COSINE"})
    db.expr = ""

    result = db.search_embedding([0.1, 0.2], k=3, payload_profile=PayloadProfile.VECTOR)

    assert result == [1]
    assert captured["output_fields"] == ["vector"]
```

- [ ] **Step 3: Run the tests and verify only the early-validation test fails**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_cloud_payload_case.py::test_case_runner_rejects_unsupported_payload_before_dataset_prepare \
  tests/test_milvus.py::test_milvus_vector_payload_requests_vector_field_and_returns_ids -q
```

Expected: Milvus translation passes against existing code; CaseRunner test fails because validation occurs later in runner construction.

- [ ] **Step 4: Add the early vector payload validator**

Add to `CaseRunner`:

```python
    def _validate_vector_payload_profile(self) -> None:
        if self.db is None or self.ca.label != CaseLabel.Performance or self.is_fts:
            return
        if not self.db.supports_payload_profile(self.ca.payload_profile):
            msg = f"{self.config.db_name} does not support payload_profile={self.ca.payload_profile.value}"
            raise NotImplementedError(msg)
```

Call it immediately after non-FTS DB initialization:

```python
            if self.ca.dataset.data.with_gt:
                self.ca.dataset.resolve_search_files(k=ground_truth_k, filters=self.ca.filters)
            self.init_db(drop_old)
            self._validate_vector_payload_profile()
            if self.ca.is_multitenant and self.db is not None:
```

- [ ] **Step 5: Run runtime and backend tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_cloud_payload_case.py \
  tests/test_large_topk_case.py \
  tests/test_milvus.py \
  tests/test_multitenant_case.py -q
```

Expected: all tests pass; existing runner-level capability checks remain intact.

- [ ] **Step 6: Commit runtime validation and contract test**

```bash
git add vectordb_bench/backend/task_runner.py tests/test_cloud_payload_case.py tests/test_milvus.py
git diff --cached --check
git -c user.name=jamesgao-jpg -c user.email=james.gao@zilliz.com commit -s -m "fix: reject unsupported payload profiles before load"
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py --repo . --commit HEAD
```

Expected: commit and DCO check pass.

### Task 5: Payload-Aware Result Identity and Export

**Files:**
- Modify: `tests/test_large_topk_frontend.py`
- Modify: `tests/test_models.py`
- Modify: `vectordb_bench/frontend/components/check_results/data.py:1-63`
- Modify: `vectordb_bench/restful/format_res.py:9-44`
- Modify: `vectordb_bench/results/getLeaderboardDataV2.py:27-54`

- [ ] **Step 1: Write a failing frontend non-merge test**

Change the test helper signature in `tests/test_large_topk_frontend.py`:

```python
def _case_result(
    *,
    k: int,
    qps: float,
    payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
) -> CaseResult:
    return CaseResult(
        task_config=TaskConfig(
            db=DB.Test,
            db_config=DB.Test.config_cls(db_label="same-db"),
            db_case_config=EmptyDBCaseConfig(),
            case_config=CaseConfig(
                case_id=CaseType.Performance768D100M,
                k=k,
                payload_profile=payload_profile,
            ),
        ),
        metrics=Metric(qps=qps, payload_profile=payload_profile.value),
    )
```

Add the test:

```python
def test_merge_tasks_keeps_payload_profiles_separate_for_same_k():
    merged, failed = data.mergeTasks(
        [
            _case_result(k=1_000_000, qps=10, payload_profile=PayloadProfile.IDS_ONLY),
            _case_result(k=1_000_000, qps=5, payload_profile=PayloadProfile.VECTOR),
        ]
    )

    assert failed == {}
    assert len(merged) == 2
    assert {item["payload_profile"] for item in merged} == {"ids_only", "vector"}
    assert len({item["case_name"] for item in merged}) == 2
```

- [ ] **Step 2: Write failing REST payload assertions**

Extend `test_rest_formatter_exports_large_topk_metrics` in `tests/test_models.py`:

```python
    test_result = _large_topk_test_result(
        Metric(
            serial_latency_p50=0.25,
            conc_latency_p50_list=[0.3],
            recall_at={100: 0.9},
            payload_profile="vector",
            payload_estimated_bytes_per_query=3_092_000_000,
        ),
        payload_profile=PayloadProfile.VECTOR,
    )

    formatted = format_results([test_result], task_label="large-topk")[0]

    assert formatted["payload_profile"] == "vector"
    assert formatted["payload_estimated_bytes_per_query"] == 3_092_000_000
```

Update the helper:

```python
def _large_topk_test_result(
    metric,
    payload_profile: PayloadProfile | None = None,
):
    return TestResult(
        run_id="large-topk",
        task_label="large-topk",
        results=[
            CaseResult(
                task_config=TaskConfig(
                    db=DB.Test,
                    db_config=DB.Test.config_cls(),
                    db_case_config=EmptyDBCaseConfig(),
                    case_config=CaseConfig(
                        case_id=CaseType.Performance768D100M,
                        k=1_000_000,
                        payload_profile=payload_profile,
                    ),
                ),
                metrics=metric,
            )
        ],
    )
```

- [ ] **Step 3: Run result tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_large_topk_frontend.py::test_merge_tasks_keeps_payload_profiles_separate_for_same_k \
  tests/test_models.py::test_rest_formatter_exports_large_topk_metrics -q
```

Expected: frontend result count is 1 or names collide, and REST output drops payload fields.

- [ ] **Step 4: Make frontend result names payload-aware**

Update imports in `check_results/data.py`:

```python
from vectordb_bench.backend.cases import CaseType, PerformanceCase
from vectordb_bench.backend.payload import PayloadProfile
```

Replace `getCaseResultName`:

```python
def getCaseResultName(task: CaseResult) -> str:
    case_config = task.task_config.case_config
    case = case_config.case
    details = []
    if case_config.k is not None and case_config.k != config.K_DEFAULT:
        details.append(f"K={case_config.k:,}")
    if (
        isinstance(case, PerformanceCase)
        and case.case_id != CaseType.CloudPayloadSearchCase
        and case.payload_profile != PayloadProfile.IDS_ONLY
    ):
        details.append(f"Payload={case.payload_profile.value}")
    if not details:
        return case.name
    return f"{case.name} ({', '.join(details)})"
```

- [ ] **Step 5: Retain payload fields in REST and legacy export**

Add fields to `FormatResult`:

```python
    payload_profile: str = "ids_only"
    payload_estimated_bytes_per_query: int = 0
```

Add the field to the non-streaming legacy row in `getLeaderboardDataV2.py`:

```python
                    "payload_profile": metrics.payload_profile,
```

- [ ] **Step 6: Add and run a legacy export assertion**

Add to `tests/test_models.py`:

```python
def test_legacy_leaderboard_exports_payload_profile(monkeypatch: pytest.MonkeyPatch):
    from vectordb_bench.results import getLeaderboardDataV2 as leaderboard

    captured = {}
    result = _large_topk_test_result(
        Metric(qps=1, recall=1, payload_profile="vector"),
        payload_profile=PayloadProfile.VECTOR,
    ).results[0]

    monkeypatch.setattr(leaderboard, "get_standard_2025_results", lambda: [result])
    monkeypatch.setattr(
        leaderboard,
        "save_to_json",
        lambda data, file_name: captured.setdefault(str(file_name), data),
    )

    leaderboard.main()

    performance_rows = next(rows for rows in captured.values() if rows)
    assert performance_rows[0]["payload_profile"] == "vector"
```

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_large_topk_frontend.py \
  tests/test_models.py -q
```

Expected: all tests pass, including old-result compatibility tests.

- [ ] **Step 7: Commit result identity and export**

```bash
git add \
  vectordb_bench/frontend/components/check_results/data.py \
  vectordb_bench/restful/format_res.py \
  vectordb_bench/results/getLeaderboardDataV2.py \
  tests/test_large_topk_frontend.py \
  tests/test_models.py
git diff --cached --check
git -c user.name=jamesgao-jpg -c user.email=james.gao@zilliz.com commit -s -m "feat: separate performance results by payload"
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py --repo . --commit HEAD
```

Expected: commit and DCO check pass.

### Task 6: Documentation, Full Verification, and PR Readiness

**Files:**
- Modify: `README.md:946-961`
- Update if implementation differs: `docs/superpowers/specs/2026-08-04-performance-payload-profiles-design.md`
- Use without committing: `/tmp/vdbbench-large-topk-payload-impact.json`

- [ ] **Step 1: Document the two return scenarios**

Add after the LAION large-topK backend note in `README.md`:

````markdown
##### Performance Response Payloads

Every vector search performance case supports an IDs-only response or a response that also includes each result vector. IDs only remains the default. Run the scenarios separately from the CLI:

```bash
vectordbbench milvusautoindex --case-type Performance768D100M --k 1000000 --payload-profile ids_only
vectordbbench milvusautoindex --case-type Performance768D100M --k 1000000 --payload-profile vector
```

The frontend can select one or both scenarios for Milvus and Zilliz Cloud. Each scenario produces independent P99 latency, QPS, and recall metrics. `qps` remains the highest observed QPS among the configured concurrency levels; VDBBench does not discover a backend concurrency limit.
````

- [ ] **Step 2: Run formatting and focused tests**

Run:

```bash
.venv/bin/python -m black --check \
  vectordb_bench/models.py \
  vectordb_bench/cli/cli.py \
  vectordb_bench/frontend/config/dbCaseConfigs.py \
  vectordb_bench/frontend/components/run_test/caseSelector.py \
  vectordb_bench/backend/task_runner.py \
  vectordb_bench/frontend/components/check_results/data.py \
  vectordb_bench/restful/format_res.py \
  vectordb_bench/results/getLeaderboardDataV2.py \
  tests/test_models.py \
  tests/test_cloud_payload_case.py \
  tests/test_large_topk_cli.py \
  tests/test_large_topk_frontend.py \
  tests/test_milvus.py

.venv/bin/python -m ruff check \
  vectordb_bench/models.py \
  vectordb_bench/cli/cli.py \
  vectordb_bench/frontend/config/dbCaseConfigs.py \
  vectordb_bench/frontend/components/run_test/caseSelector.py \
  vectordb_bench/backend/task_runner.py \
  vectordb_bench/frontend/components/check_results/data.py \
  vectordb_bench/restful/format_res.py \
  vectordb_bench/results/getLeaderboardDataV2.py \
  tests/test_models.py \
  tests/test_cloud_payload_case.py \
  tests/test_large_topk_cli.py \
  tests/test_large_topk_frontend.py \
  tests/test_milvus.py

.venv/bin/python -m pytest \
  tests/test_models.py \
  tests/test_cloud_payload_case.py \
  tests/test_cloud_payload_search.py \
  tests/test_cloud_cold_latency_case.py \
  tests/test_case_runner_reuse.py \
  tests/test_large_topk_case.py \
  tests/test_large_topk_cli.py \
  tests/test_large_topk_frontend.py \
  tests/test_milvus.py \
  tests/test_milvus_zilliz_cli.py \
  tests/test_multitenant_case.py \
  tests/test_turbopuffer_cli.py -q
```

Expected: Black and Ruff exit 0; all focused tests pass.

- [ ] **Step 3: Run repository CI parity checks**

Run:

```bash
make lint
make unittest
```

Expected: the same lint and deterministic unit-test targets used by `.github/workflows/pull_request.yml` pass.

- [ ] **Step 4: Rescan and validate the impact map**

Run:

```bash
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/impact_scan.py rescan \
  --repo /home/ubuntu/largeTopk/VectorDBBench \
  --map /tmp/vdbbench-large-topk-payload-impact.json \
  --base origin/main

python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/impact_scan.py validate \
  --map /tmp/vdbbench-large-topk-payload-impact.json
```

Expected: no unmapped consumers and validation passes. Inspect and disposition any newly reported file before continuing.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-04-performance-payload-profiles-design.md
git diff --cached --check
git -c user.name=jamesgao-jpg -c user.email=james.gao@zilliz.com commit -s -m "docs: document performance payload profiles"
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py --repo . --commit HEAD
```

Expected: commit succeeds. If the design spec did not change, stage and commit only `README.md`.

- [ ] **Step 6: Verify all outgoing commits and worktree state**

Run:

```bash
git log --format='%h %s%n%(trailers:key=Signed-off-by,valueonly)' origin/LargeTopk..HEAD
python3 /home/ubuntu/.codex/skills/vdbbench-dev/scripts/check_dco.py \
  --repo /home/ubuntu/largeTopk/VectorDBBench \
  --range origin/LargeTopk..HEAD
git status --short --branch
```

Expected: every outgoing commit has `jamesgao-jpg <james.gao@zilliz.com>`, and the worktree is clean.

- [ ] **Step 7: Push and verify the upstream branch**

The user previously selected `zilliztech/VectorDBBench` and `LargeTopk` as the destination.

Run:

```bash
git push origin HEAD:LargeTopk
git ls-remote origin refs/heads/LargeTopk
git rev-parse HEAD
```

Expected: the remote `LargeTopk` SHA exactly matches local `HEAD`.

- [ ] **Step 8: Review PR #834 description against the completed implementation**

Open [zilliztech/VectorDBBench#834](https://github.com/zilliztech/VectorDBBench/pull/834) and ensure it describes:

- first-class payload support across vector `PerformanceCase` workloads;
- IDs-only and vector scenarios;
- Milvus and Zilliz Cloud frontend scope;
- P99, QPS, recall, and payload-aware result identity;
- no concurrency-limit discovery;
- no `query_mode` change;
- focused tests and backend probe status.

Expected: the PR description matches the branch. If authenticated GitHub tooling is unavailable, report that the branch was pushed but the PR description could not be updated from this environment.
