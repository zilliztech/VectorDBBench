# FTS E2E Results

This directory stores full-text-search end-to-end benchmark artifacts. The layout is grouped first by backend, then by dataset family:

```text
fts_results/
  milvus/
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

Each benchmark run should be stored as a timestamped run bundle under the matching backend and dataset directory:

```text
fts_results/<backend>/<dataset>/<YYYYMMDD-HHMMSS>-<dataset-size>/
  metadata.yaml
  server-deployment.md
  client-command.txt
  result.json
  run.log
  notes.md
```

Use `metadata.yaml` for machine class, backend version, dataset size, benchmark label, git revision, and whether the server was freshly torn down before the run. Store raw VectorDBBench result JSON as `result.json`; keep `notes.md` for caveats such as competing processes, disk pressure, failed retries, or non-default backend tuning.

Do not commit credentials, API tokens, private addresses, SSH keys, or `hosts.yaml`.
