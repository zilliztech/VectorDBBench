# TiDB SPFresh

Use `vectordbbench tidbspfresh` to benchmark a TiDB deployment with SPFresh
support. The existing `vectordbbench tidb` command continues to use TiFlash.
Both commands use the same connection options and the `tidb` dependency extra:

```sh
pip install 'vectordb-bench[tidb]'
```

The SPFresh client requires a compatible TiDB and worker deployment supporting
`USING SPFRESH`, `VECTOR_INDEX_PARAM`, and
`information_schema.TIDB_SPFRESH_INDEX_STATUS`. It supports L2 and cosine
distance, unfiltered ANN searches, `id >= ...` filters, and equality filters on
the `labels` column. Label cases create `labels VARCHAR(64) NOT NULL` and include
it in the vector index with `STORING (labels)`.

## Load and search

This example runs the standard Cohere 1M performance case with top-100 queries.
Create the database first and substitute the connection details for your deployment.
The load stage replaces the benchmark table in that database.

```sh
vectordbbench tidbspfresh \
  --host 127.0.0.1 --port 4000 --username root --password '' \
  --db-name cohere_1m --case-type Performance768D1M --k 100 \
  --spfresh-vector-index-param 'min_partition_size=64,max_partition_size=512,write_beam_size=8,search_beam_size=174,oversample_factor=1,hamming_factor=96,rabitq4_factor=4'
```

These are example index settings, not a recall guarantee. Validate recall for
your dataset, filter, and top-k. The client passes `VECTOR_INDEX_PARAM` through
to TiDB; it does not supply dataset-specific tuning defaults.

| Option | Default | Behavior |
| --- | --- | --- |
| `--spfresh-build-mode` | `inline` | `inline` creates the index with the empty table; `non-inline` creates it after loading all rows. |
| `--spfresh-vector-index-param` | Unset | Raw index parameters supplied only when creating the index. |
| `--spfresh-wait-timeout` | `600` seconds | Maximum checkpoint polling time after loading; the case's existing optimize timeout still applies to the whole optimize stage. |

Add `--spfresh-build-mode non-inline` to the example to load rows before creating
the index. Both modes use the standard concurrent insert runner, including
`--insert-batch-size` and `--load-concurrency`. The client also splits each batch
across up to ten writer connections, as the existing TiDB client does.

After the runner joins all insert workers, the client obtains a fresh TiDB TSO.
It waits until the index reports `READY`, `is_ready=1`, and
`applied_base_ts >= TSO` before the search stages begin. Non-inline mode creates
the index before checking this condition. Missing/invalid status, an index that
needs rebuilding, query errors, and checkpoint timeouts fail the benchmark.

Results use the existing insert, optimize, load, recall, and QPS metrics and are
identified as `TiDBSPFresh`. There are no SPFresh-specific timing fields.

## Filtered search

Use the existing `NewIntFilterPerformanceCase` for numeric filters or
`LabelFilterPerformanceCase` for labels. For example, replace the case selection
in the load command with:

```sh
--case-type LabelFilterPerformanceCase \
--dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
--label-percentage 0.01
```

Use explicitly calibrated index parameters for filtered cases. Search-only label
cases require an existing table and SPFresh index created with the label column.

## Reusing an index

`--skip-load` reuses the table and index; it skips both index creation and
optimization. `--spfresh-vector-index-param` and `--spfresh-build-mode` do not
modify the existing index in this mode. Supply mutable search settings with SQL
before running a search-only benchmark, for example:

```sql
USE cohere_1m;
ALTER TABLE vector_bench_test
  ALTER INDEX idx_embedding_spfresh_cosine
  SET VECTOR_INDEX_PARAM 'search_beam_size=174,oversample_factor=1,hamming_factor=96,rabitq4_factor=4';
```

Then run:

```sh
vectordbbench tidbspfresh \
  --host 127.0.0.1 --port 4000 --username root --password '' \
  --db-name cohere_1m --case-type Performance768D1M --k 100 --skip-load
```

The index name is `idx_embedding_spfresh_cosine` or `idx_embedding_spfresh_l2`,
depending on the dataset metric. Creation parameters such as partition sizes
and `write_beam_size` require a new index. This client does not provide a
build-only or split-load workflow, delete benchmarks, or compatibility with
obsolete SPFresh development indexes.

## YAML batches

Use `tidbspfresh` as the command key and the same option names with underscores:

```yaml
tidbspfresh:
  - host: 127.0.0.1
    port: 4000
    username: root
    password: ''
    db_name: cohere_1m
    case_type: Performance768D1M
    k: 100
    spfresh_build_mode: non-inline
    spfresh_vector_index_param: 'min_partition_size=64,max_partition_size=512,write_beam_size=8,search_beam_size=174,oversample_factor=1,hamming_factor=96,rabitq4_factor=4'
```
