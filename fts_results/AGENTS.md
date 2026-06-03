# FTS Results Reporting Contract

These instructions apply to everything under `fts_results/`.

## Required Layout

Each backend/dataset leaf directory must use this layout:

```text
fts_results/<backend>/<dataset>/
  report.md
  raw_results/
    <VectorDBBench result JSON files>
```

Use `.gitkeep` only when `raw_results/` is otherwise empty. Commit every raw
VectorDBBench JSON that supports a reported result before or with the report
change. Do not summarize a run unless its raw JSON is present in `raw_results/`.

The root `fts_results/report.md` is the cross-backend master comparison report.
It is intentionally not a backend/dataset leaf report and does not use the
backend-specific report template below.

## Required Report Structure

Every `report.md` must keep these top-level sections in this exact order and
with these exact headings:

```markdown
# <Backend> <Dataset> FTS E2E Report

## Prequsites

### Physical Machine Stats

## Server Setup

## VDBBench Running

## Result
```

The spelling `Prequsites` is intentionally preserved to match the current
requested report contract. Do not rename it unless all reports and this file
are updated together.

## Section Requirements

- `Prequsites`: record backend, dataset family, dataset size for each result,
  run date, raw result path, source runbook, and any caveats.
- `Physical Machine Stats`: record CPU, memory, disk quota, OS, and role for
  the client and server machines. For managed backends, record the client
  machine and state that server-side resources are managed externally.
- `Server Setup`: include a shell script that can reproduce the server or
  service setup. Record image tags, container memory limits, JVM heap settings,
  volume names, namespace names, and config deviations.
- `VDBBench Running`: include the exact shell script for running the benchmark
  from the client. Use placeholders for private hosts, credentials, tokens, and
  key paths. The command must include task label, case type, dataset-with-size
  type, stages, k, concurrency list, duration, and timeout.
- For new FTS E2E runs, explicitly use
  `--num-concurrency "1,10,20,40,60,80"` to probe the saturation point. Keep
  historical report commands unchanged when documenting already-committed raw
  results.
- `Result`: include a Markdown table summarizing every committed raw result in
  `raw_results/`. At minimum include raw JSON filename, task label, dataset
  size, load duration, QPS, recall, NDCG, MRR, p95, p99, and concurrent QPS.

## Safety

Never commit credentials, PEM paths, private host addresses, API tokens, config
files containing secrets, or `hosts.yaml`. Raw VectorDBBench JSONs may be
committed only after confirming secret fields are masked.
