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
