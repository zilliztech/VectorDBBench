# Cloud Leaderboard Results

This directory is reserved for Cloud Leaderboard result assets.

The current VDBBench result pages recursively scan `vectordb_bench/results/**/result_*.json`.
Do not add raw `result_*.json` files here until the Cloud Leaderboard frontend has a dedicated
ingestion path or the default result collector explicitly excludes this directory.

Raw case results should be staged under each case's `raw_results/` directory once that case's
frontend support is implemented.

