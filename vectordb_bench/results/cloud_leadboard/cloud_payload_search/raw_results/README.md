# Search Raw Results Archive

This directory stores raw VDBBench JSON outputs for the single-tenant 100M
search matrix. The Markdown report summarizes results for humans; this archive
keeps the source artifacts needed to import results into an official VDBBench
cloud leaderboard.

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
| `product` | `zilliz_cloud_tiered_4cu`, `zilliz_cloud_capacity_12cu`, `pinecone_serverless`, `turbopuffer_unpinned`, `turbopuffer_pinned` |
| `filter_type` | `unfiltered`, `int_filter`, `scalar_label_filter` |
| `filter_rate` | `na` for unfiltered, otherwise labels such as `1p`, `0_5p`, `20p` |
| `payload_profile` | `ids_only`, `scalar_label`, `vector` |
| `phase` | `serial_recall`, `concurrent_qps` |

## Manifest

`manifest.jsonl` has one JSON object per raw result file. It is the machine
index for ingestion. Do not add lines for missing or planned files.

Required fields:

| Field | Meaning |
|---|---|
| `case_id` | Stable matrix key: `<product>__<filter_type>__<filter_rate>__<payload_profile>` |
| `product` | Product directory name |
| `filter_type` | Filter mode directory name |
| `filter_rate` | Filter-rate directory name |
| `payload_profile` | Payload profile directory name |
| `phase` | `serial_recall` or `concurrent_qps` |
| `raw_json` | Path to the raw JSON, relative to repo root |
| `run_id` | VDBBench run id |
| `db_label` | VDBBench run label |
| `framework_repo` | Source framework repository |
| `framework_branch` | Source framework branch |
| `framework_commit` | Source framework commit |
| `framework_state` | `clean` or a concise description of local changes |

Raw result JSONs must be copied exactly as VDBBench emitted them. Credentials
must remain redacted in the JSON before committing.

## Verification

Run this before committing any report or raw-result update:

```bash
python3 cloud_payload_search/verify_search_results.py
```

The verifier checks that:

- Every manifest raw JSON exists.
- Manifest `run_id`, `db_label`, and payload profile match the raw JSON.
- Every manifest entry is referenced by `cloud_payload_search/single_tenant_100m_search.md`.
- Every raw JSON path in the report exists in the manifest.
- Reported recall, NDCG, QPS, latency, and payload bytes match the raw JSON at
  the precision shown in the Markdown table.
