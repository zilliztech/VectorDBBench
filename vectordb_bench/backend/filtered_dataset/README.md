# Filtered-search fixtures

Generates synthetic datasets for filtered search with controlled metadata distributions and
exact ground truth, so filtered-search results can be reproduced and compared across
selectivities, skew and vector/metadata correlation. Runner integration is not part of this
module.

## Generate and verify

```shell
python -m vectordb_bench.backend.filtered_dataset generate --config spec.yaml --out ./filtered_fixture
python -m vectordb_bench.backend.filtered_dataset verify ./filtered_fixture
```

`generate` prints each filter case with its target and actual selectivity. `verify` re-evaluates
every filter from the written metadata and recomputes the nearest neighbors with an independent
exact search (`--sample N` checks the distances for N random queries only). Without `--config`
the defaults below are used.

## Spec

```yaml
name: filtered_synthetic
seed: 42
num_train: 100000        # ids are 0..num_train-1
num_test: 1000
dim: 128
metric_type: L2          # L2, COSINE or IP
num_clusters: 32         # vectors are a Gaussian mixture
cluster_std: 0.25
k: 100                   # ground-truth width
labels:                  # one categorical value per row -> column `labels`
  num_values: 100
  distribution: zipf     # uniform or zipf
  zipf_s: 1.1
  correlation: 0.0       # 0 = independent of the vectors, 1 = follows vector position
int_attr:                # one integer per row in [low, high] -> column `int_attr`
  low: 0
  high: 9999
  distribution: uniform
  correlation: 0.0
cases:
  - {type: label_equal, target_selectivity: 0.01}
  - {type: int_range, target_selectivity: 0.1, start_quantile: 0.0}
```

Selectivity is the fraction of rows that match a filter (`filter_rate` in VectorDBBench is
`1 - selectivity`). Each case becomes the closest filter the data allows: `label_equal` picks
the label value whose frequency is closest to the target, and `int_range` picks a half-open
range `[low, high)` starting at `start_quantile`. The achieved selectivity is recorded, and is
never zero.

Metadata is assigned by ranking rows on a score that mixes a random projection of the vector
with noise, so the marginal distribution is exact while `correlation` sets how closely the
metadata follows vector position.

## Output

```
train.parquet                  id, emb
test.parquet                   id, emb
scalar_labels.parquet          id, labels, int_attr   (row i describes train id i)
neighbors.parquet              id, neighbors_id       unfiltered top-k
neighbors_<case_id>.parquet    id, neighbors_id       top-k over the rows that match the case
manifest.json
```

The layout is the one the custom-dataset reader already loads. Ground truth is ranked by
(distance, id), with distances computed in float64, so equal distances are broken by the
smaller id. A query gets fewer than `k` neighbors when fewer rows match.

`manifest.json` stores the full spec and, for each case:

- `expr` and `condition`
- `target_selectivity`, `actual_selectivity`, `filter_rate` and `matching_rows`
- `min_groundtruth_width` and `queries_with_fewer_than_k`
- `queries_with_boundary_ties`: queries whose k-th and (k+1)-th distances are equal, so
  another top-k is equally valid
- `unfiltered_topk_pass_rate`: the share of each query's unfiltered top-k that passes
- `queries_with_no_unfiltered_match`, next to the count expected if the metadata were
  independent of the vectors. Compare the two to see how much the metadata is correlated
  with vector position.
- `same_condition_as`, set when two targets resolve to the same filter

The same spec and seed always produce identical files.
