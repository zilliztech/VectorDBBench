# Milvus E2E result observations

The Milvus/Zilliz performance path adds optional native metrics without changing
`insert_duration`, `optimize_duration`, `load_duration` or QPS semantics:

| Field | Meaning |
|---|---|
| `compaction_duration` | Seconds accumulated around each compact request and its completion wait, including normal compaction and force-merge retries. Null unless the compaction sequence completed. |
| `compaction_status` | `completed`, `skipped` (GPU or no plan), `failed` (including denied operation), or `not_recorded` (old result/other backend/no optimization). |
| `output_fields` | Resolved extra fields requested by vector search: `[]` for IDs only, the vector field, or scalar label field. Null when not recorded. |

Compaction duration excludes flush, pre-compaction sorting/index readiness,
inter-attempt sleeps, plan inspection, final index wait and load refresh. It is
client-observed elapsed time, not CPU time or server-reported background work.
The optimization worker returns the observation to its parent explicitly.
Existing exception handling is preserved; callers can distinguish a denied
compaction from a completed one even when native search continues.

Dataset metadata is emitted for built-in datasets too: name, vector count (`size`),
dimension (`dim`), metric, source and known revision. Existing richer dataset
metadata is preserved. Missing dimensions for nonvector workloads remain null.
Native serialization and result collection retain the fields; old JSON initializes
compaction as `not_recorded`/null. No old optimize duration is relabeled.

Backend contract: Milvus/ZillizCloud, inspected at parent `9782fde`; tests use mocked
PyMilvus 2.6.11 calls and a real local worker process. The existing compact request,
completion and force-merge coverage behavior is unchanged. No service version is
claimed validated and no live backend probe or performance run was performed.
Arbitrary output-field combinations are outside this change.

Level-1 verification:

```sh
python -m pytest -q tests/test_e2e_metadata.py tests/test_milvus.py::TestMilvusOptimize
```

Expected: targeted metadata, serialization, worker, timing and optimize tests pass.

## Opt-in stage events and compaction segment counts

The E2E orchestrator can set `E2E_STAGE_EVENTS=1`, `E2E_RUN_ID`, `E2E_CASE_ID`
and `E2E_ATTEMPT`. Benchmark workers emit bounded atomic stdout lines prefixed
`E2E_STAGE ` followed by versioned JSON. A unique span ID links `start`, `end`
or `failed` for insertion, each actual compaction RPC and wait, serial search,
and each concurrent-search setting. Events carry UTC and monotonic timestamps;
search events include concurrency. Compaction worker stdout is inherited by the
parent runner. Other VDBBench usage emits no stage events by default.

The E2E orchestrator can additionally opt in with `E2E_COLLECTION_RELOAD=1`.
The Milvus/Zilliz Cloud optimization then emits one `index_compaction` parent
span around flush, index waits and compaction. Child `index` and `compaction`
spans include `parent_span_id`; their durations are observations and must not
be added to infer the parent duration. Afterward, `collection_unload` measures
release through `NotLoad`, and `collection_load` measures load through `Loaded`;
the events share a `reload_id`. A failed or timed-out release or load fails the
case. This explicit reload changes the workload and does not guarantee a cold
cache. With the option absent, the existing refresh-load behavior is retained.

With events enabled, each compaction also samples persistent segment counts just
before and just after the compaction span. The `segments` observation and native
`compaction_segments` metric retain both timestamps and mark an unavailable
read explicitly. These observations add two bounded metadata reads and are not
part of the measured compaction duration. They do not count segments for historical
results or promise that background segment state cannot change between the reads.

The consumer's field definitions and retention policy are described in the
[orchestrator telemetry contract](https://github.com/jamesgao-jpg/milvus-benchmark-orchestrator/blob/codex/orchestrator-mvp/collector/CONTRACT.md).
