# Performance Payload Profiles Design

## Status

- Design: approved on 2026-08-04
- Implementation: not started
- Related issue: [zilliztech/VectorDBBench#826](https://github.com/zilliztech/VectorDBBench/issues/826)
- Target branch: `LargeTopk`
- Baseline revision: `4ef433dae4b3a4b8a7af52dca107d01832bd4f4a`

## Problem

VDBBench already knows how to ask a backend for IDs only or additional response payload, but ordinary vector performance cases cannot select that behavior through a first-class `CaseConfig` option. Payload selection is currently concentrated in specialized cases such as `CloudPayloadSearchCase`, which duplicates the ordinary performance-case shape instead of treating response payload as an execution option.

Large-topK runs make this distinction important. Returning 1M IDs and returning 1M IDs plus 1M vectors exercise materially different response sizes, so latency and throughput must be recorded as separate benchmark results even when dataset, index, K, and concurrency settings are identical.

## Verified Existing Behavior

The following statements are verified against the baseline revision:

- `Case.payload_profile` already defaults to `ids_only`, and every vector `PerformanceCase` inherits it: [`cases.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/backend/cases.py#L100-L161).
- `CaseConfig` currently exposes `case_id`, `custom_case`, K, and concurrency settings, but no top-level payload field: [`models.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/models.py#L211-L248).
- Serial and multiprocessing search runners already pass non-default payload profiles and reject profiles that the backend does not support: [`serial_runner.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/backend/runner/serial_runner.py#L140-L183), [`mp_runner.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/backend/runner/mp_runner.py#L48-L104).
- Milvus declares vector-payload support and sets its vector field in `output_fields`; VDBBench then extracts IDs for metric calculation: [`milvus.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/backend/clients/milvus/milvus.py#L442-L495).
- Zilliz Cloud inherits the Milvus client implementation: [`zilliz_cloud.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/backend/clients/zilliz_cloud/zilliz_cloud.py#L1-L26).
- Existing performance metrics already include QPS, serial P99, per-concurrency P99, recall, and payload metadata: [`metric.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/metric.py#L14-L48).
- Frontend result grouping currently distinguishes K but not payload profile, so otherwise identical IDs-only and vector runs can overwrite or merge: [`data.py`](https://github.com/zilliztech/VectorDBBench/blob/4ef433dae4b3a4b8a7af52dca107d01832bd4f4a/vectordb_bench/frontend/components/check_results/data.py#L9-L63).

## Goals

1. Make response payload a first-class option for every vector search case whose instantiated case is a `PerformanceCase`.
2. Support two ordinary performance return scenarios in this change:
   - `ids_only`
   - `vector`
3. Allow IDs-only and vector scenarios to be run separately or together without creating new `CaseType` values.
4. Keep P99 latency, QPS, and recall results separate and payload-aware.
5. Provide implementation and acceptance coverage for Milvus and Zilliz Cloud.
6. Preserve legacy payload configuration and existing specialized cloud cases.

## Non-Goals

- Do not remove, rename, or refactor `CloudPayloadSearchCase` in this change.
- Do not add a new large-topK case class. Large-topK remains a parameterized `Performance768D100M` run.
- Do not discover or report a backend's highest sustainable concurrency.
- Do not change the configured concurrency list or concurrency timeout behavior.
- Do not change recall, NDCG, ground-truth, or latency algorithms introduced by the existing large-topK work.
- Do not configure Milvus `query_mode=large_topk`; that work remains outside this change.
- Do not add new payload support implementations for other backends.
- Do not retain returned vectors in result files or benchmark process state after IDs are extracted.

## Scope

The shared option applies to every vector case that resolves to a `PerformanceCase`, including:

- standard performance cases;
- fixed int-filter performance cases;
- generated int-filter performance cases;
- label-filter performance cases;
- custom-dataset performance cases;
- existing `PerformanceCase`-based cloud search cases.

It does not newly apply to capacity, streaming, insert, cold-latency, or full-text-search cases. Their existing payload behavior remains unchanged.

## Configuration Contract

`CaseConfig` gains an optional top-level field:

```python
payload_profile: PayloadProfile | None = None
```

Examples:

```python
CaseConfig(
    case_id=CaseType.Performance768D100M,
    k=1_000_000,
    payload_profile=PayloadProfile.IDS_ONLY,
)

CaseConfig(
    case_id=CaseType.Performance768D100M,
    k=1_000_000,
    payload_profile=PayloadProfile.VECTOR,
)
```

`None` is intentional rather than an explicit `ids_only` model default:

- old serialized results that lack the field continue to load;
- legacy `custom_case={"payload_profile": ...}` remains authoritative when no top-level value is present;
- case classes retain their existing default behavior, which is IDs only for ordinary performance cases.

Resolution rules are deterministic:

1. An explicitly provided top-level value is valid only when `case_id` resolves to a `PerformanceCase`; otherwise reject it as a configuration error.
2. When only the top-level value exists, copy it into the case-constructor arguments.
3. When only legacy `custom_case.payload_profile` exists, preserve it.
4. When both exist and normalize to the same `PayloadProfile`, accept the configuration.
5. When both exist and differ, reject the configuration with a validation error.

The source `custom_case` dictionary must not be mutated during case construction.

Adding the field to `CaseConfig` also makes IDs-only and vector configs produce different `CaseConfig` hashes. The collection load-reuse key should remain unchanged for these two profiles because requesting a returned vector does not change the stored collection schema.

## CLI Contract

The existing `--payload-profile` option remains the single CLI entry point. It will be passed into top-level `CaseConfig.payload_profile` for ordinary vector performance cases.

Examples:

```bash
vectordbbench milvusautoindex --case-type Performance768D100M --k 1000000 --payload-profile ids_only
vectordbbench milvusautoindex --case-type Performance768D100M --k 1000000 --payload-profile vector
```

Existing specialized cloud and FTS mappings continue to populate their legacy constructor data for compatibility. If the CLI supplies both paths, they will contain the same value and pass conflict validation.

The CLI continues to expose the existing complete `PayloadProfile` choice set because specialized cases use additional profiles. The support guarantee for ordinary vector performance cases in this change is limited to `ids_only` and `vector`.

## Frontend Contract

Every selectable vector `PerformanceCase` item gains a `Return scenario` multiselect with:

- `IDs only`, selected by default;
- `Vector payload`.

Selecting both expands each base `CaseConfig` into two independent configs before task generation. This avoids duplicating case registrations and ensures each scenario has its own timing and metric record.

The vector option is presented in this iteration only when every active backend is Milvus or Zilliz Cloud. IDs-only behavior remains available for all existing backends. Mixed backend selections containing another backend therefore remain IDs-only through this new control.

An empty return-scenario selection blocks that case from submission and displays a validation error. Capacity, streaming, and FTS UI entries do not receive this control.

## Execution Flow

```text
CLI / batch / frontend
  -> CaseConfig.payload_profile
  -> CaseConfig resolves legacy and top-level values
  -> instantiated PerformanceCase.payload_profile
  -> CaseRunner validates backend capability before dataset load
  -> SerialSearchRunner and MultiProcessingSearchRunner
  -> backend search request includes the selected payload profile
  -> backend response is fully received
  -> VDBBench extracts IDs
  -> recall and latency/QPS metrics are calculated
  -> result is serialized with payload identity
```

The capability check should happen immediately after database client initialization and before dataset preparation or loading. Existing runner checks remain as defense in depth. An unsupported vector profile must fail before an expensive dataset load begins.

For vector payload, the response vector contributes to backend processing, network transfer, client decoding, latency, and QPS. VDBBench intentionally discards the vector after extracting result IDs because recall only needs IDs and retaining up to 1M vectors would create avoidable memory pressure.

## Metric Semantics

No metric formulas or units change.

| Field | Meaning |
|---|---|
| `serial_latency_p99` | P99 wall-clock latency across the serial query sample; raw result value remains in seconds. |
| `conc_latency_p99_list` | P99 latency in seconds for each configured concurrency level. |
| `qps` | Highest successful QPS observed among the configured concurrency levels. |
| `conc_num_list` / `conc_qps_list` | Configured concurrency levels and their observed successful QPS. |
| `recall` | Mean recall at the requested K for the serial query sample. |
| `recall_at` | Existing multi-cutoff recall values available from the large-topK implementation. |
| `payload_profile` | Requested response shape, such as `ids_only` or `vector`. |
| `payload_estimated_bytes_per_query` | Existing deterministic estimate, not measured network bytes. |

There is no `highest_concurrency_achieved` field. A failed or throttled configured concurrency keeps the existing runner behavior and does not introduce automatic concurrency discovery.

## Result Identity and Serialization

Payload profile becomes part of every result's logical identity:

```text
database + database label + case + K + payload profile
```

Frontend display names append a payload suffix for ordinary performance cases, for example:

```text
Search Performance Test (100M Dataset, 768 Dim) (K=1,000,000, Payload=vector)
```

The existing `CloudPayloadSearchCase` name already contains its profile and must not receive a duplicate suffix.

Required serialization behavior:

- `CaseConfig` JSON includes the top-level field when explicitly selected.
- metric JSON continues to include `payload_profile` and `payload_estimated_bytes_per_query`.
- REST `FormatResult` explicitly declares both payload fields so Pydantic does not discard them.
- legacy leaderboard export includes `payload_profile` to avoid ambiguous duplicate rows.
- old result files missing top-level payload data load as the existing IDs-only default unless legacy custom-case data specifies another profile.

## Backend Contract

### Context

- Backends: Milvus and Zilliz Cloud
- Deployment versions: unknown until benchmark execution
- SDK requirement: `pymilvus>=2.6.15,<3.0.0` in the baseline `pyproject.toml`
- VDBBench revision: `4ef433dae4b3a4b8a7af52dca107d01832bd4f4a`

### Capabilities

| Capability | Intended semantics | VDBBench translation | Evidence | Probe | Status |
|---|---|---|---|---|---|
| IDs only | Search returns IDs without requested vector fields. | Milvus uses `output_fields=None`; runners omit the payload argument for the default path. | Baseline Milvus and runner source linked above. | Not run | VERIFIED in source and mocked tests |
| Vector payload | Search requests each hit's vector while VDBBench extracts IDs for metrics. | Milvus uses `output_fields=[vector_field]`; Zilliz Cloud inherits Milvus. | Baseline Milvus and Zilliz Cloud source linked above. | Not run | VERIFIED translation; deployment behavior unprobed |
| Unsupported profile | Reject before expensive dataset loading. | Check `supports_payload_profile()` after client initialization; retain runner checks. | Existing capability methods and runner source linked above. | Not run | Design requirement |

### Unsupported Combinations

- The frontend does not offer the new vector scenario for active backend sets outside Milvus and Zilliz Cloud.
- Ordinary vector performance cases do not gain a `text` payload mode.
- Actual large-topK vector-payload readiness is not established until a target Milvus and Zilliz Cloud functional probe succeeds.

### Remaining Assumption

LIKELY: target Milvus and Zilliz Cloud deployments will honor the existing `output_fields=[vector_field]` translation at the requested K. This must be verified with a small authorized functional probe before claiming benchmark readiness; implementation unit tests alone do not prove deployment behavior.

## Compatibility

- Existing `CaseConfig` JSON without `payload_profile` remains valid.
- Existing `custom_case.payload_profile` remains valid.
- Explicit top-level payload configuration on a non-`PerformanceCase` is rejected rather than silently ignored.
- Existing `CloudPayloadSearchCase`, `CloudColdLatencyCase`, `CloudMultiTenantSearchCase`, and FTS behavior remains unchanged.
- IDs-only names remain unchanged where possible; non-default vector results receive an explicit suffix.
- Existing backend capability methods remain the authority for runtime support.
- Existing result artifacts are not regenerated.
- No dependency changes are required.

## Error Handling

- Conflicting top-level and legacy profiles: configuration validation error.
- Top-level payload profile on a non-`PerformanceCase`: configuration validation error.
- Empty frontend profile selection: submission validation error.
- Backend reports the profile unsupported: `NotImplementedError` before dataset preparation/loading.
- Backend search fails or times out: preserve existing runner retry, failure, and timeout behavior.
- Returned IDs are insufficient for requested K: preserve existing large-topK validation and metric behavior.

## Verification Plan

Implementation will follow test-driven development with these focused checks:

1. `CaseConfig`
   - default construction remains IDs only;
   - explicit top-level vector construction;
   - legacy-only construction;
   - matching dual specification;
   - conflicting dual specification;
   - top-level profile rejected for non-`PerformanceCase` case IDs;
   - serialization, deserialization, and hash separation;
   - no mutation of `custom_case`.
2. CLI
   - ordinary performance case maps `--payload-profile vector` to top-level `CaseConfig`;
   - IDs-only default remains compatible;
   - help text describes ordinary vector performance use;
   - existing cloud and FTS mappings remain valid.
3. Frontend
   - all vector `PerformanceCase` items support IDs-only and vector expansion;
   - selecting both creates two distinct `CaseConfig` objects;
   - capacity, streaming, and FTS cases are unchanged;
   - unsupported or mixed active backend sets do not offer vector through the new control.
4. Runtime
   - unsupported vector profile fails before dataset preparation/loading;
   - existing serial and concurrent runners receive the resolved profile;
   - Milvus vector mode sets the vector output field and still returns IDs.
5. Results
   - same DB/case/K with different profiles remains two frontend results;
   - QPS/recall/table views use the same payload-aware identity;
   - REST and legacy export include payload fields;
   - old result files continue to load.
6. Regression
   - focused large-topK, payload, Milvus, CLI, frontend, and model tests;
   - lint and repository CI unit-test target;
   - impact-map rescan and validation after implementation.

No live performance benchmark is part of implementation verification. A functional backend probe, if authorized later, establishes request/response semantics only and is not performance evidence.

## Planned Implementation Surfaces

- `vectordb_bench/models.py`
- `vectordb_bench/cli/cli.py`
- `vectordb_bench/frontend/config/dbCaseConfigs.py`
- `vectordb_bench/frontend/components/run_test/caseSelector.py`
- `vectordb_bench/frontend/components/check_results/data.py`
- `vectordb_bench/backend/task_runner.py`
- `vectordb_bench/restful/format_res.py`
- `vectordb_bench/results/getLeaderboardDataV2.py`
- focused existing test modules
- `README.md`

No case registry, backend registry, payload enum, metric formula, dataset artifact, or dependency file is expected to change.
