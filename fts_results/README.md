# FTS E2E Results

This directory stores full-text-search end-to-end benchmark artifacts. The layout is grouped first by backend, then by dataset family:

```text
fts_results/
  Milvus/
    msmarco/
    hotpotqa/
  ElasticSearch/
    msmarco/
    hotpotqa/
  Vespa/
    msmarco/
    hotpotqa/
  Turbopuffer/
    msmarco/
    hotpotqa/
```

Each backend/dataset leaf directory should store a single report and the raw result files that support it:

```text
fts_results/<backend>/<dataset>/
  report.md
  raw_results/
    <VectorDBBench result JSON files>
```

Follow `fts_results/AGENTS.md` for the exact report structure. The raw JSON files in `raw_results/` are the source of truth; each `report.md` summarizes those files and records the reproducible server setup plus VDBBench command.

Do not commit credentials, API tokens, private addresses, SSH keys, or `hosts.yaml`.
