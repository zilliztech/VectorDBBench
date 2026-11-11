import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from vectordb_bench.backend.cases import PerformanceCustomDataset
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.filtered_dataset import (
    CaseType,
    Distribution,
    FilterCaseSpec,
    FilteredDatasetSpec,
    IntSpec,
    LabelSpec,
    generate_dataset,
    groundtruth,
    verify_dataset,
)
from vectordb_bench.backend.filtered_dataset.__main__ import cli
from vectordb_bench.backend.filtered_dataset.generator import _exact_counts, _probabilities, build_cases


def small_spec(**overrides) -> FilteredDatasetSpec:
    fields = {
        "num_train": 2_000,
        "num_test": 25,
        "dim": 8,
        "num_clusters": 4,
        "k": 10,
        "labels": LabelSpec(num_values=12, distribution=Distribution.ZIPF),
        "int_attr": IntSpec(low=0, high=99),
        "cases": [
            FilterCaseSpec(type=CaseType.LABEL_EQUAL, target_selectivity=0.02),
            FilterCaseSpec(type=CaseType.LABEL_EQUAL, target_selectivity=0.2),
            FilterCaseSpec(type=CaseType.INT_RANGE, target_selectivity=0.003),
            FilterCaseSpec(type=CaseType.INT_RANGE, target_selectivity=0.3, start_quantile=0.5),
        ],
    }
    fields.update(overrides)
    return FilteredDatasetSpec(**fields)


def reference_topk(queries, train, mask, k, metric_type):
    """Full sort by (distance, id) with the plain (non-expanded) distance formula."""
    ids = np.flatnonzero(mask)
    out = []
    for q in queries:
        if metric_type == MetricType.L2:
            dist = ((train[ids] - q) ** 2).sum(axis=1)
        else:
            dist = -(train[ids] @ q)
        out.append(ids[np.lexsort((ids, dist))][:k])
    return out


@pytest.mark.parametrize("metric_type", [MetricType.L2, MetricType.COSINE, MetricType.IP])
def test_generated_fixture_verifies(tmp_path, metric_type):
    manifest = generate_dataset(small_spec(metric_type=metric_type), tmp_path)

    assert verify_dataset(tmp_path) == []
    for case in manifest["cases"]:
        assert (tmp_path / case["groundtruth_file"]).exists()
        assert case["matching_rows"] > 0
        assert case["actual_selectivity"] == pytest.approx(case["matching_rows"] / 2_000)


def test_groundtruth_ids_satisfy_their_filters(tmp_path):
    manifest = generate_dataset(small_spec(), tmp_path)
    labels = pq.read_table(tmp_path / "scalar_labels.parquet").to_pydict()
    for case in manifest["cases"]:
        rows = pq.read_table(tmp_path / case["groundtruth_file"]).column("neighbors_id").to_pylist()
        cond = case["condition"]
        for row in rows:
            for i in row:
                if cond["op"] == "==":
                    assert labels["labels"][i] == cond["value"]
                else:
                    assert cond["low"] <= labels["int_attr"][i] < cond["high_exclusive"]


def test_same_seed_gives_identical_files(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    generate_dataset(small_spec(), first)
    generate_dataset(small_spec(), second)
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_different_seed_changes_the_data(tmp_path):
    generate_dataset(small_spec(seed=1), tmp_path / "a")
    generate_dataset(small_spec(seed=2), tmp_path / "b")
    assert (tmp_path / "a" / "train.parquet").read_bytes() != (tmp_path / "b" / "train.parquet").read_bytes()


def test_exact_topk_matches_full_sort_and_breaks_ties_by_id():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 3, size=(40, 2)).astype(np.float64)  # few distinct points, so many ties
    train = np.concatenate([base] * 25)  # 1000 rows, every point repeated
    queries = rng.integers(0, 3, size=(30, 2)).astype(np.float64)
    mask = rng.random(len(train)) < 0.6

    for metric_type in (MetricType.L2, MetricType.IP):
        result = groundtruth.exact_topk(queries, train, mask, k=7, metric_type=metric_type)
        expected = reference_topk(queries, train, mask, 7, metric_type)
        for got, want in zip(result.ids, expected, strict=True):
            np.testing.assert_array_equal(got, want)
        assert result.boundary_ties > 0


def test_exact_topk_is_independent_of_chunk_size(monkeypatch):
    rng = np.random.default_rng(1)
    train = rng.integers(0, 4, size=(500, 3)).astype(np.float64)
    queries = rng.standard_normal((9, 3))
    mask = np.ones(len(train), dtype=bool)
    whole = groundtruth.exact_topk(queries, train, mask, 15, MetricType.L2)
    monkeypatch.setattr(groundtruth, "TRAIN_CHUNK", 7)
    monkeypatch.setattr(groundtruth, "QUERY_BATCH", 2)
    chunked = groundtruth.exact_topk(queries, train, mask, 15, MetricType.L2)
    for a, b in zip(whole.ids, chunked.ids, strict=True):
        np.testing.assert_array_equal(a, b)
    assert whole.boundary_ties == chunked.boundary_ties


def test_fewer_matches_than_k_and_empty_filters():
    train = groundtruth.prepare_vectors(np.eye(6), MetricType.L2)
    queries = train[:2]
    mask = np.zeros(6, dtype=bool)
    mask[[1, 4]] = True
    result = groundtruth.exact_topk(queries, train, mask, k=5, metric_type=MetricType.L2)
    assert [len(row) for row in result.ids] == [2, 2]
    assert set(result.ids[0]) == {1, 4}

    empty = groundtruth.exact_topk(queries, train, np.zeros(6, dtype=bool), k=5, metric_type=MetricType.L2)
    assert [len(row) for row in empty.ids] == [0, 0]


@pytest.mark.parametrize("distribution", [Distribution.UNIFORM, Distribution.ZIPF])
def test_label_counts_follow_the_distribution_exactly(distribution):
    probabilities = _probabilities(7, distribution, 1.3)
    counts = _exact_counts(probabilities, 1_001)
    assert counts.sum() == 1_001
    assert np.all(np.abs(counts - probabilities * 1_001) < 1)


def test_correlation_moves_matching_rows_away_from_unrelated_queries(tmp_path):
    def no_match_count(correlation):
        spec = small_spec(
            labels=LabelSpec(num_values=12, distribution=Distribution.ZIPF, correlation=correlation),
            int_attr=IntSpec(low=0, high=99, correlation=correlation),
            num_test=200,
        )
        manifest = generate_dataset(spec, tmp_path / str(correlation))
        return sum(case["queries_with_no_unfiltered_match"] for case in manifest["cases"])

    assert no_match_count(1.0) > no_match_count(0.0)


def test_int_range_is_never_empty_when_a_heavy_value_sits_at_the_start():
    spec = small_spec(
        int_attr=IntSpec(low=0, high=999, distribution=Distribution.ZIPF, zipf_s=1.5),
        cases=[FilterCaseSpec(type=CaseType.INT_RANGE, target_selectivity=0.05)],
    )
    n = spec.num_train
    labels = np.zeros(n, dtype=np.int64)
    int_attr = np.zeros(n, dtype=np.int64)
    int_attr[n // 2 :] = np.arange(n - n // 2)  # half the rows share the value 0
    [(case, mask)] = build_cases(spec, labels, ["label_0"], int_attr)
    assert mask.sum() > 0
    assert case.condition["low"] == 0


def test_cases_that_resolve_to_the_same_filter_are_flagged(tmp_path):
    spec = small_spec(
        labels=LabelSpec(num_values=4, distribution=Distribution.UNIFORM),
        cases=[
            FilterCaseSpec(type=CaseType.LABEL_EQUAL, target_selectivity=0.2),
            FilterCaseSpec(type=CaseType.LABEL_EQUAL, target_selectivity=0.3),
        ],
    )
    first, second = generate_dataset(spec, tmp_path)["cases"]
    assert first["same_condition_as"] is None
    assert second["same_condition_as"] == first["case_id"]


def test_verify_reports_a_tampered_groundtruth(tmp_path):
    manifest = generate_dataset(small_spec(), tmp_path)
    case = manifest["cases"][0]
    path = tmp_path / case["groundtruth_file"]
    table = pq.read_table(path).to_pydict()
    labels = pq.read_table(tmp_path / "scalar_labels.parquet").column("labels").to_pylist()
    outsider = next(i for i, value in enumerate(labels) if value != case["condition"]["value"])
    table["neighbors_id"][0][0] = outsider
    pq.write_table(pa.Table.from_pydict(table, schema=pq.read_schema(path)), path)

    problems = verify_dataset(tmp_path)
    assert any("do not satisfy the filter" in p for p in problems)


def test_verify_reports_a_wrong_selectivity(tmp_path):
    generate_dataset(small_spec(), tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][1]["matching_rows"] += 1
    manifest_path.write_text(json.dumps(manifest))
    assert any("matching rows" in p for p in verify_dataset(tmp_path))


def test_refuses_to_overwrite_a_non_empty_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        generate_dataset(small_spec(), tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"metric_type": MetricType.HAMMING},
        {"int_attr": {"low": 10, "high": 5}},
        {"labels": {"num_values": 5_000}},
        {"cases": [{"type": "label_equal", "target_selectivity": 0}]},
    ],
)
def test_invalid_specs_are_rejected(overrides):
    with pytest.raises(ValidationError):
        small_spec(**overrides)


def test_cli_generate_then_verify(tmp_path):
    config = tmp_path / "spec.yaml"
    config.write_text(
        "num_train: 500\nnum_test: 5\ndim: 4\nk: 5\nlabels: {num_values: 6}\n"
        "cases:\n  - {type: label_equal, target_selectivity: 0.1}\n"
    )
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--config", str(config), "--out", str(out), "--seed", "3"])
    assert result.exit_code == 0, result.output
    assert json.loads((out / "manifest.json").read_text())["spec"]["seed"] == 3

    result = runner.invoke(cli, ["verify", str(out)])
    assert result.exit_code == 0, result.output
    assert "OK: 1 cases verified" in result.output


def test_fixture_loads_through_the_custom_dataset_reader(tmp_path):
    spec = small_spec(num_train=600, num_test=4, k=100)
    generate_dataset(spec, tmp_path)
    case = PerformanceCustomDataset(
        name="generated",
        description="",
        load_timeout=1,
        optimize_timeout=1,
        dataset_config={
            "name": "generated",
            "dir": str(tmp_path),
            "size": spec.num_train,
            "dim": spec.dim,
            "metric_type": "L2",
            "file_count": 1,
            "with_gt": True,
        },
    )
    dataset = case.dataset
    dataset.prepare(filters=non_filter, with_scalar_labels=True)

    assert len(dataset.test_data) == spec.num_test
    assert len(dataset.gt_data) == spec.num_test
    ids = np.concatenate([batch["id"].to_numpy() for batch in dataset])
    np.testing.assert_array_equal(ids, np.arange(spec.num_train))
    # The runners look labels up by train id, so row i must describe id i.
    written = pq.read_table(tmp_path / "scalar_labels.parquet").column("labels").to_pylist()
    assert dataset.scalar_labels["labels"][[0, 7, 599]].to_list() == [written[0], written[7], written[599]]
