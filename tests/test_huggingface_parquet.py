from pathlib import Path

import polars as pl
import pytest

from vectordb_bench import config
from vectordb_bench.backend.clients import MetricType
from vectordb_bench.backend.data_source import DatasetSource, HuggingFaceReader
from vectordb_bench.backend.dataset import (
    ParquetDataset,
    ParquetDatasetManager,
    get_dataset_manager,
    get_registered_datasets,
)
from vectordb_bench.backend.filter import LabelFilter, non_filter
from vectordb_bench.models import DatasetMetadata


def test_hugging_face_reader_expands_wildcard_selectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = tmp_path / "snapshot"
    for name in (
        "data/neighbors.parquet",
        "data/test-00001-of-00002.parquet",
        "data/test-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
        "data/train-00000-of-00002.parquet",
    ):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    paths = HuggingFaceReader().read(
        "VDBBench/example",
        ["data/train-*.parquet", "data/test-*.parquet", "data/neighbors.parquet"],
        tmp_path / "cache",
        revision="revision",
    )

    assert list(paths) == [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
        "data/test-00000-of-00002.parquet",
        "data/test-00001-of-00002.parquet",
        "data/neighbors.parquet",
    ]
    assert calls == [
        {
            "repo_id": "VDBBench/example",
            "repo_type": "dataset",
            "revision": "revision",
            "cache_dir": tmp_path / "cache",
            "allow_patterns": ["data/train-*.parquet", "data/test-*.parquet", "data/neighbors.parquet"],
        }
    ]


def test_hugging_face_parquet_manager_resolves_roles_and_concatenates_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config, "DATASET_LOCAL_DIR", tmp_path / "datasets")
    files = {
        "train/part-1.parquet": pl.DataFrame({"id": [2, 3], "emb": [[2.0, 2.0], [3.0, 3.0]]}),
        "train/part-0.parquet": pl.DataFrame({"id": [0, 1], "emb": [[0.0, 0.0], [1.0, 1.0]]}),
        "test/part-1.parquet": pl.DataFrame({"id": [12], "emb": [[12.0, 12.0]]}),
        "test/part-0.parquet": pl.DataFrame({"id": [10, 11], "emb": [[10.0, 10.0], [11.0, 11.0]]}),
        "neighbors/neighbors.parquet": pl.DataFrame({"id": [10, 11, 12], "neighbors": [[0, 1], [1, 2], [2, 3]]}),
    }
    resolved = {}
    for name, frame in files.items():
        path = tmp_path / "snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        resolved[name] = path

    class Reader:
        def read(self, *args, **kwargs):
            return resolved

    monkeypatch.setattr(DatasetSource, "reader", lambda _source: Reader())
    manager = ParquetDatasetManager(
        data=ParquetDataset(
            name="tiny-parquet",
            size=4,
            dim=2,
            metric_type=MetricType.IP,
            use_shuffled=False,
            with_gt=True,
            source=DatasetSource.HuggingFace,
            source_dataset="VDBBench/example",
            source_revision="revision",
            train_selectors=("train/*.parquet",),
            query_selectors=("test/*.parquet",),
            gt_selector="neighbors/neighbors.parquet",
            gt_neighbors_field="neighbors",
            ground_truth_width=2,
            query_count=3,
            family="VDBBench",
            dataset_metadata={"normalization": "l2"},
        )
    )

    assert manager.prepare(k=2)
    assert manager.train_files == ["train/part-0.parquet", "train/part-1.parquet"]
    assert manager.test_data == [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]
    assert [row.tolist() for row in manager.gt_data.iter_rows()] == [[0, 1], [1, 2], [2, 3]]
    assert [item for batch in manager.iter_batches(2) for item in batch["id"].tolist()] == [0, 1, 2, 3]
    assert manager.result_metadata["train_files"] == manager.train_files
    assert manager.result_metadata["query_files"] == ["test/part-0.parquet", "test/part-1.parquet"]
    metadata = DatasetMetadata.model_validate(manager.result_metadata).model_dump(mode="json")
    assert metadata["storage_format"] == "parquet"
    assert metadata["normalization"] == "l2"


def test_vdbbench_multimodal_datasets_are_registered_for_performance_and_ui():
    expected = {
        "multimodal-embedding-1m": (
            "VDBBench/multimodal-embedding-1M",
            ("train.parquet",),
            ("test.parquet",),
            "neighbors.parquet",
        ),
        "multimodal-embedding-10m": (
            "VDBBench/multimodal-embedding-10M",
            ("data/train-*.parquet",),
            ("data/test-*.parquet",),
            "data/neighbors.parquet",
        ),
        "multimodal-embedding-100m": (
            "VDBBench/multimodal-embedding-100M",
            ("train/shard-*/*.parquet",),
            ("test/*.parquet",),
            "neighbors/neighbors.parquet",
        ),
    }
    managers = get_registered_datasets(family="VDBBench")
    assert {manager.data.name for manager in managers} == set(expected)

    for name, (repository, train, query, ground_truth) in expected.items():
        manager = get_dataset_manager(name)
        assert isinstance(manager, ParquetDatasetManager)
        assert manager.data.source == DatasetSource.HuggingFace
        assert manager.data.source_dataset == repository
        assert manager.data.train_selectors == train
        assert manager.data.query_selectors == query
        assert manager.data.gt_selector == ground_truth
        assert manager.data.dim == 4096
        assert manager.data.metric_type == MetricType.IP
        assert manager.data.ground_truth_width == 100
        assert manager.data.query_count == 10_000
