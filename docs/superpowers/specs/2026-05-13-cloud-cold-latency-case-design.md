# Cloud Cold Latency Case Design

Date: 2026-05-13

## Goal

Add a cloud leaderboard case that measures cold-search latency and immediately repeated warm-search latency for the same query workload.

The first version targets cloud benchmark runs where collections may be loaded but not recently searched. It reports how much slower the first pass is than the immediate second pass.

The canonical workload is:

- Run 1000 serial searches and record cold latency stats.
- Immediately run the same 1000 serial searches again and record warm latency stats.
- Report first-query, p99, p95, and average latency for both passes.
- Report cold-to-warm ratios for each latency statistic.

## Case Shape

Add a new case named `CloudColdLatencyCase`.

Default parameters:

- `dataset_with_size_type`: `None`, which means LAION 100M, matching `CloudPayloadSearchCase`.
- `payload_profile`: `ids_only`.
- `filter_rate`: `None`.
- `label_percentage`: `None`.
- `query_count`: `1000`.

The case should support the same cloud search envelope knobs as `CloudPayloadSearchCase`:

- Unfiltered search: omit both `filter_rate` and `label_percentage`.
- Int-filter search: set `filter_rate`.
- Scalar-label search: set `label_percentage`.
- Response payload shape: set `payload_profile` to `ids_only`, `vector`, or `scalar_label`.

`filter_rate` and `label_percentage` are mutually exclusive. A run that needs unfiltered and 90% filtered results should create separate task configs, for example one unfiltered task and one task with `label_percentage=0.9`.

## Architecture

Add a dedicated `CaseLabel.CloudColdLatency` so `CaseRunner.run()` can route this case without overloading normal performance behavior.

`CloudColdLatencyCase` should live near `CloudPayloadSearchCase` in `vectordb_bench/backend/cases.py`. It should reuse the same dataset, filter, payload-profile, timeout, and validation conventions where possible.

Add a dedicated runner named `ColdWarmSearchRunner` under `vectordb_bench/backend/runner/`. This runner owns the cold/warm two-pass behavior and returns latency-only stats.

This avoids changing the public return contract of `SerialSearchRunner`, which currently returns recall, NDCG, p99, and p95 for normal performance cases.

## Data Flow

1. The task config selects `CaseType.CloudColdLatencyCase`.
2. The case resolves dataset, filter, payload profile, and query count.
3. `CaseRunner._pre_run()` prepares the dataset using the existing source path. Training files remain tied to `TaskStage.LOAD`; test vectors and filter files are prepared for search.
4. `_run_cloud_cold_latency_case()` initializes search embeddings using the same normalization path as normal performance cases.
5. `ColdWarmSearchRunner` takes the first `query_count` test vectors.
6. The runner opens one `db.init()` context, calls `db.prepare_filter()` once, and runs the cold serial pass.
7. Without closing the context, sleeping, or optimizing, the runner immediately runs the warm serial pass over the same vectors.
8. Each pass records per-query latencies and computes first-query, p99, p95, and average latency in seconds.
9. Cold-to-warm ratios are computed as cold stat divided by warm stat.
10. The final JSON is stored in `Metric.additional_parameters["cold_latency"]`.

## Result Shape

Use stable snake_case keys:

```json
{
  "cold_stats": {
    "first_query_latency": 0.0,
    "p99_latency": 0.0,
    "p95_latency": 0.0,
    "avg_latency": 0.0
  },
  "warm_stats": {
    "first_query_latency": 0.0,
    "p99_latency": 0.0,
    "p95_latency": 0.0,
    "avg_latency": 0.0
  },
  "cold_warm_ratio": {
    "first_query_latency_ratio": 0.0,
    "p99_latency_ratio": 0.0,
    "p95_latency_ratio": 0.0,
    "avg_latency_ratio": 0.0
  }
}
```

`Metric.payload_profile` and `Metric.payload_estimated_bytes_per_query` should be populated the same way they are for performance cloud search cases.

The first implementation does not need new top-level `Metric` fields. Frontend work can later promote these stats into charts or tables after result files are available.

## Runner Behavior

The runner should use the same low-level search semantics as existing serial search:

- Validate `db.supports_payload_profile(payload_profile)` before queries start.
- Pass `payload_profile` to `search_embedding()` unless the profile is `ids_only`.
- Use the case filter through `db.prepare_filter()`.
- Use the existing search retry policy.
- Keep query order fixed between cold and warm passes.

The runner does not compute recall or NDCG. This case is latency-only.

If `query_count` is larger than the available test vectors, fail clearly instead of cycling through vectors. Cold/warm comparisons should use one fixed query set.

## Error Handling

Case validation:

- Reject both `filter_rate` and `label_percentage` being set in one case.
- Reject `query_count <= 0`.
- Normalize string payload profiles into `PayloadProfile`.
- Normalize string dataset values into `DatasetWithSizeType`.

Runner validation:

- Fail fast for unsupported payload profiles.
- Fail if there are fewer than `query_count` test vectors.
- For ratio calculation, return `0.0` if the warm denominator is `0`. Real DB latency should not be zero; this primarily protects fake tests and malformed data.

## CLI And Config

Add the case to the existing CLI custom-case pattern.

Expected custom case parameters:

- `payload_profile`
- `cloud_filter_rate`
- `cloud_label_percentage`
- `dataset_with_size_type`
- `cloud_cold_query_count`

The CLI should map `cloud_filter_rate` to `filter_rate` and `cloud_label_percentage` to `label_percentage`, matching the existing `CloudPayloadSearchCase` naming.

Frontend case selection can be added after the backend result shape is validated. CLI support is enough for the first implementation.

## Testing

Add focused unit tests:

- `CloudColdLatencyCase` defaults to LAION 100M, `ids_only`, and `query_count=1000`.
- The case accepts custom dataset, payload profile, int filter, and label filter.
- The case rejects both filter types in one config.
- The case rejects invalid query counts.
- `CaseConfig` builds `CloudColdLatencyCase` from `custom_case`.
- The runner computes cold stats, warm stats, and ratios from a fake DB.
- The runner passes payload profile into both cold and warm searches.
- The runner fails fast for unsupported payload profiles.
- `CaseRunner` stores the cold latency JSON under `Metric.additional_parameters["cold_latency"]`.
- Existing performance cases still use `SerialSearchRunner` unchanged.

## Out Of Scope

- Frontend chart or table integration.
- Recall and NDCG for cold/warm runs.
- Concurrent cold/warm latency.
- Sleeping, unloading, or externally forcing a cold cache state.
- Running int filter and scalar-label filter in the same case config.
- Backend-specific cold-cache reset APIs.
