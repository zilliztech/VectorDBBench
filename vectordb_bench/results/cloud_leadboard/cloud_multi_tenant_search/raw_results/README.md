# Multitenant Raw Results Archive

This directory stores raw VDBBench JSON outputs for the CloudMultiTenantSearchCase matrix. The Markdown report summarizes results for humans; this archive keeps the source artifacts needed to import results into an official VDBBench cloud leaderboard.

## Layout

```text
raw_results/
  manifest.jsonl
  <product>/
    <filter_type>/
      <filter_rate>/
        <payload_profile>/
          <phase>/
            result_<date>_<run_id>_<product>.json
```

Path components:

| Component | Values |
|---|---|
| `product` | `zilliz_cloud_tiered_1cu`, `zilliz_cloud_capacity_2cu`, `turbopuffer`, `pinecone_serverless` |
| `filter_type` | `unfiltered`, `int_filter`, `scalar_label_filter` |
| `filter_rate` | `na` for unfiltered; actual `--cloud-filter-rate` labels such as `99_9p`, `99p`, `90p`, `50p`; scalar-label percentage labels such as `0_1p`, `5p`, `50p` |
| `payload_profile` | `ids_only`, `scalar_label`, `vector` |
| `phase` | `concurrent_qps` |

For integer filters, `filter_rate` is the actual VDBBench CLI threshold used by `--cloud-filter-rate`, so `99_9p` means `id >= int(dataset_size * 0.999)` and leaves roughly 0.1% of rows as candidates. The manifest retains `source_rate_label` for the original run-label token.

Pinecone Serverless entries are preserved from a detached c4-only run. Their raw JSON metrics use `conc_num_list=[4]`, while the Zilliz and Turbopuffer rows use the main `60,80` concurrency sweep.

## Manifest

`manifest.jsonl` has one JSON object per raw result file. It is the machine index for ingestion. Do not add lines for missing or planned files.

Required fields:

| Field | Meaning |
|---|---|
| `case_id` | Stable matrix key: `<product>__<filter_type>__<filter_rate>__<payload_profile>` |
| `product` | Product directory name |
| `filter_type` | Filter mode directory name |
| `filter_rate` | Filter-rate directory name |
| `filter_rate_display` | Human-readable filter rate |
| `filter_rate_value` | Numeric filter rate used by the case, or null for unfiltered |
| `source_rate_label` | Original rate token embedded in the VDBBench run label |
| `payload_profile` | Payload profile directory name |
| `phase` | `concurrent_qps` |
| `raw_json` | Path to the raw JSON, relative to repo root |
| `run_id` | VDBBench run id |
| `db_label` | VDBBench run label |
| `framework_repo` | Source framework repository |
| `framework_branch` | Source framework branch |
| `framework_commit` | Source framework commit |
| `framework_state` | `clean` or a concise description of local changes |
