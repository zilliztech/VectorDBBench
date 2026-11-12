import json
import logging
import shutil
import threading
from contextlib import contextmanager

import click
import numpy as np
import pyarrow.parquet as pq
import pytest

from vectordb_bench.backend.assembler import Assembler, FilterNotSupportedError
from vectordb_bench.backend.cases import CaseType, FilteredFixtureCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import VectorDB
from vectordb_bench.backend.clients.milvus.config import FLATConfig, MilvusConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.clients.test.config import TestConfig, TestIndexConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import FilterOp, FixtureLabelFilter, IntRangeFilter, LabelFilter
from vectordb_bench.backend.filtered_dataset import (
    CaseType as FixtureCaseType,
)
from vectordb_bench.backend.filtered_dataset import (
    FilterCaseSpec,
    FilteredDatasetSpec,
    LabelSpec,
    generate_dataset,
)
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.models import CaseConfig, TaskConfig, TaskStage

LABEL_CASE = "00_label_equal_0.01"
RANGE_CASE = "01_int_range_0.02"
WIDE_LABEL_CASE = "02_label_equal_0.2"


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixture")
    spec = FilteredDatasetSpec(
        num_train=2_000,
        num_test=10,
        dim=8,
        num_clusters=4,
        labels=LabelSpec(num_values=20),
        cases=[
            FilterCaseSpec(type=FixtureCaseType.LABEL_EQUAL, target_selectivity=0.01),
            FilterCaseSpec(type=FixtureCaseType.INT_RANGE, target_selectivity=0.02),
            FilterCaseSpec(type=FixtureCaseType.LABEL_EQUAL, target_selectivity=0.2),
        ],
    )
    generate_dataset(spec, out)
    return out


def manifest_case(fixture_dir, case_id):
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    return next(case for case in manifest["cases"] if case["case_id"] == case_id)


def test_label_case_maps_to_a_label_filter_with_the_manifest_value(fixture_dir):
    case = FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id=LABEL_CASE)
    entry = manifest_case(fixture_dir, LABEL_CASE)

    filters = case.filters
    assert isinstance(filters, FixtureLabelFilter)
    assert filters.type == FilterOp.StrEqual
    assert filters.label_field == "labels"
    assert filters.label_value == entry["condition"]["value"]
    assert filters.groundtruth_file == entry["groundtruth_file"]
    assert case.filter_rate == pytest.approx(entry["filter_rate"])
    assert case.dataset.data.size == 2_000
    assert entry["expr"] in case.description


def test_range_case_maps_to_an_int_range_filter(fixture_dir):
    case = FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id=RANGE_CASE)
    condition = manifest_case(fixture_dir, RANGE_CASE)["condition"]

    filters = case.filters
    assert isinstance(filters, IntRangeFilter)
    assert (filters.type, filters.int_field) == (FilterOp.NumRange, "int_attr")
    assert (filters.low, filters.high_exclusive) == (condition["low"], condition["high_exclusive"])
    assert case.with_scalar_labels


def test_unknown_case_and_wrong_version_are_rejected(fixture_dir, tmp_path):
    with pytest.raises(ValueError, match="has no case 'nope'"):
        FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id="nope")

    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    manifest["format_version"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported format_version"):
        FilteredFixtureCase(fixture_dir=str(tmp_path), fixture_case_id=LABEL_CASE)


def test_case_config_builds_the_fixture_case(fixture_dir):
    custom = get_custom_case_config(
        {"case_type": "FilteredFixtureCase", "fixture_dir": str(fixture_dir), "fixture_case": RANGE_CASE}
    )
    case = CaseConfig(case_id=CaseType.FilteredFixtureCase, custom_case=custom).case
    assert isinstance(case, FilteredFixtureCase)
    assert case.fixture_case_id == RANGE_CASE

    with pytest.raises(click.UsageError):
        get_custom_case_config({"case_type": "FilteredFixtureCase", "fixture_dir": None, "fixture_case": None})


class OrdinaryLabelFilter(LabelFilter):
    """A plain label filter pointed at a given ground-truth file (allows_short_groundtruth stays False)."""

    gt_file_name: str

    @property
    def groundtruth_file(self) -> str:
        return self.gt_file_name


def test_short_groundtruth_loads_for_fixture_filters_only(fixture_dir):
    case = FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id=LABEL_CASE)
    matching = manifest_case(fixture_dir, LABEL_CASE)["matching_rows"]
    assert matching < 100

    case.dataset.prepare(filters=case.filters, with_scalar_labels=True)
    assert case.dataset.gt_data.width == matching

    # The same narrow file is still rejected for an ordinary filter that expects K neighbors.
    narrow = OrdinaryLabelFilter(label_percentage=0.01, gt_file_name=case.filters.groundtruth_file)
    with pytest.raises(ValueError, match="smaller than requested K"):
        case.dataset.prepare(filters=narrow, with_scalar_labels=True)


class RecordingDB(VectorDB):
    """Answers each query with a fixed list of ids and records the filter it was given."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prepared = None

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        self.prepared = filters

    def insert_embeddings(self, embeddings, metadata, **kwargs):
        return len(metadata), None

    def search_embedding(self, query, k=100, **kwargs):
        return self.answers.pop(0)

    def optimize(self, data_size=None):
        pass


def test_recall_is_scored_against_the_neighbors_the_filter_allows(fixture_dir):
    case = FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id=LABEL_CASE)
    case.dataset.prepare(filters=case.filters, with_scalar_labels=True)
    truth = [row.tolist() for row in case.dataset.gt_data.iter_rows()]

    perfect = RecordingDB(truth)
    runner = SerialSearchRunner(perfect, case.dataset.test_data, case.dataset.gt_data, k=100, filters=case.filters)
    recall, ndcg, *_ = runner.search((runner.test_data, runner.ground_truth))
    assert (recall, ndcg) == (1.0, 1.0)
    assert perfect.prepared is not None
    assert perfect.prepared.label_value == case.filters.label_value

    half = RecordingDB(row[: len(row) // 2] for row in truth)
    runner = SerialSearchRunner(half, case.dataset.test_data, case.dataset.gt_data, k=100, filters=case.filters)
    recall, *_ = runner.search((runner.test_data, runner.ground_truth))
    assert recall == pytest.approx(np.mean([(len(row) // 2) / len(row) for row in truth]), abs=1e-4)


def test_insert_batches_carry_the_range_column(fixture_dir):
    case = FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id=RANGE_CASE)
    case.dataset.prepare(filters=case.filters, with_scalar_labels=True)
    runner = ConcurrentInsertRunner(
        RecordingDB([]), case.dataset, normalize=False, filters=case.filters, with_scalar_labels=True
    )
    runner._iter_lock = threading.Lock()
    runner._deadline = None
    runner._dataset_iter = case.dataset.iter_batches(runner.batch_size)

    batch = runner._next_vector_batch()
    written = pq.read_table(fixture_dir / "scalar_labels.parquet").column("int_attr").to_pylist()
    assert batch["int_data"]["int_attr"] == [written[i] for i in batch["metadata"]]
    assert batch["labels_data"] is not None

    label_case = FilteredFixtureCase(fixture_dir=str(fixture_dir), fixture_case_id=LABEL_CASE)
    label_case.dataset.prepare(filters=label_case.filters, with_scalar_labels=True)
    runner.filters = label_case.filters
    runner.dataset = label_case.dataset
    runner._dataset_iter = label_case.dataset.iter_batches(runner.batch_size)
    assert "int_data" not in runner._next_vector_batch()


def test_milvus_builds_the_range_expression():
    milvus = Milvus.__new__(Milvus)
    milvus._is_fts = False
    milvus._range_filter_field = "int_attr"
    milvus.prepare_filter(
        IntRangeFilter(low=4, high_exclusive=113, filter_rate=0.98, gt_file_name="n.parquet", matching_rows=51)
    )
    assert milvus.expr == "int_attr >= 4 and int_attr < 113"
    assert FilterOp.NumRange in Milvus.supported_filter_types

    milvus._range_filter_field = None
    with pytest.raises(ValueError, match="no range field"):
        milvus.prepare_filter(
            IntRangeFilter(low=0, high_exclusive=1, filter_rate=0.5, gt_file_name="n.parquet", matching_rows=1)
        )


def _task(db, db_config, db_case_config, fixture_dir, case_id):
    return TaskConfig(
        db=db,
        db_config=db_config,
        db_case_config=db_case_config,
        case_config=CaseConfig(
            case_id=CaseType.FilteredFixtureCase,
            custom_case={"fixture_dir": str(fixture_dir), "fixture_case_id": case_id},
        ),
        stages=[TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_SERIAL],
    )


def test_assembler_skips_fixture_cases_a_client_cannot_run(fixture_dir, caplog):
    tasks = [
        _task(DB.Milvus, MilvusConfig(uri="http://localhost:19530"), FLATConfig(), fixture_dir, RANGE_CASE),
        _task(DB.Test, TestConfig(), TestIndexConfig(), fixture_dir, LABEL_CASE),
        _task(DB.Test, TestConfig(), TestIndexConfig(), fixture_dir, RANGE_CASE),
    ]
    with caplog.at_level(logging.WARNING):
        runner = Assembler.assemble_all("run-id", "task-label", tasks, DatasetSource.S3)

    assert [(r.config.db, r.ca.fixture_case_id) for r in runner.case_runners] == [(DB.Milvus, RANGE_CASE)]
    skipped = [rec.getMessage() for rec in caplog.records if "Skipping" in rec.getMessage()]
    assert len(skipped) == 2
    assert any("Label filters" in msg for msg in skipped)
    assert any("NumRange filters" in msg and "int_attr >=" in msg for msg in skipped)


def test_assembler_still_fails_ordinary_unsupported_cases():
    task = TaskConfig(
        db=DB.Test,
        db_config=TestConfig(),
        db_case_config=TestIndexConfig(),
        case_config=CaseConfig(
            case_id=CaseType.LabelFilterPerformanceCase,
            custom_case={"dataset_with_size_type": "Small Cohere (768dim, 100K)", "label_percentage": 0.01},
        ),
        stages=[TaskStage.SEARCH_SERIAL],
    )
    with pytest.raises(FilterNotSupportedError):
        Assembler.assemble_all("run-id", "task-label", [task], DatasetSource.S3)


def test_a_range_case_does_not_reuse_a_label_case_load(fixture_dir):
    tasks = [
        _task(DB.Milvus, MilvusConfig(uri="http://localhost:19530"), FLATConfig(), fixture_dir, LABEL_CASE),
        _task(DB.Milvus, MilvusConfig(uri="http://localhost:19530"), FLATConfig(), fixture_dir, RANGE_CASE),
        _task(DB.Milvus, MilvusConfig(uri="http://localhost:19530"), FLATConfig(), fixture_dir, WIDE_LABEL_CASE),
    ]
    runners = Assembler.assemble_all("run-id", "task-label", tasks, DatasetSource.S3).case_runners
    keys = {r.ca.fixture_case_id: r.load_reuse_key() for r in runners}

    # Label cases share one load; the range case needs a collection with the int column.
    assert keys[LABEL_CASE] == keys[WIDE_LABEL_CASE]
    assert keys[RANGE_CASE] != keys[LABEL_CASE]


def test_k_above_the_fixture_k_is_rejected(tmp_path):
    spec = FilteredDatasetSpec(
        num_train=1_000,
        num_test=5,
        dim=8,
        k=10,
        labels=LabelSpec(num_values=4),
        cases=[FilterCaseSpec(type=FixtureCaseType.LABEL_EQUAL, target_selectivity=0.3)],
    )
    generate_dataset(spec, tmp_path)
    case_id = json.loads((tmp_path / "manifest.json").read_text())["cases"][0]["case_id"]
    case = FilteredFixtureCase(fixture_dir=str(tmp_path), fixture_case_id=case_id)
    assert case.fixture_case["matching_rows"] > 100

    # 10-wide ground truth cannot score K=100 when far more than 100 rows match.
    with pytest.raises(ValueError, match="smaller than requested K"):
        case.dataset.prepare(filters=case.filters, with_scalar_labels=True, k=100)
    case.dataset.prepare(filters=case.filters, with_scalar_labels=True, k=10)
    assert case.dataset.gt_data.width == 10


def test_saved_result_loads_after_the_fixture_moves(fixture_dir, tmp_path, monkeypatch):
    moved = tmp_path / "fixture"
    shutil.copytree(fixture_dir, moved)
    monkeypatch.chdir(tmp_path)
    custom = get_custom_case_config(
        {"case_type": "FilteredFixtureCase", "fixture_dir": "fixture", "fixture_case": RANGE_CASE}
    )
    assert custom["fixture_dir"] == str(moved.resolve())

    shutil.rmtree(moved)
    case = CaseConfig(case_id=CaseType.FilteredFixtureCase, custom_case=custom).case
    assert case.filters.type == FilterOp.NumRange
    assert case.fixture_report["case_id"] == RANGE_CASE
