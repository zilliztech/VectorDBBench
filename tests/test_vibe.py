import json
import pickle
from inspect import isabstract
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from vectordb_bench import config
from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import Performance, type2case
from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig, MetricType
from vectordb_bench.backend.data_source import DatasetSource, HuggingFaceReader
from vectordb_bench.backend.dataset import (
    Dataset,
    DatasetManager,
    DatasetWithSizeType,
    CustomDataset,
    Hdf5Dataset,
    Hdf5DatasetManager,
    ParquetDatasetManager,
    get_registered_datasets,
)
from vectordb_bench.backend.filter import LabelFilter, non_filter
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.frontend.components.check_results.data import mergeTasks
from vectordb_bench.frontend.config.dbCaseConfigs import UI_CASE_CLUSTERS
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, CaseType, TaskConfig, TestResult
from vectordb_bench.restful.format_res import format_results


def test_vibe_is_registered_through_generic_dataset_managers():
    vibe = get_registered_datasets(family="VIBE")
    assert isabstract(DatasetManager)
    assert isinstance(Dataset.COHERE.manager(100_000), ParquetDatasetManager)
    assert len(vibe) == 24
    assert len({manager.data.name for manager in vibe}) == 24
    assert all(isinstance(manager, Hdf5DatasetManager) for manager in vibe)
    assert sum(manager.data.distribution == "id" for manager in vibe) == 15
    assert sum(manager.data.distribution == "ood" for manager in vibe) == 9
    assert all(manager.data.file_name == f"{manager.data.name}.hdf5" for manager in vibe)
    ip_datasets = [manager.data for manager in vibe if manager.data.source_distance == "ip"]
    assert len(ip_datasets) == 4
    assert all(data.metric_type == MetricType.IP for data in ip_datasets)
    assert isinstance(DatasetSource.HuggingFace.reader(), HuggingFaceReader)


def test_hugging_face_reader_uses_pinned_single_file_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = get_registered_datasets(family="VIBE")[0]
    revision = manager.data.source_revision
    destination = tmp_path / "cached.hdf5"
    destination.write_bytes(b"hdf5")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(destination)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    paths = HuggingFaceReader().read(
        "vector-index-bench/vibe",
        ["glove-200-cosine.hdf5"],
        tmp_path / "cache",
        revision=revision,
    )

    assert paths == {"glove-200-cosine.hdf5": destination}
    assert calls == [
        {
            "repo_id": "vector-index-bench/vibe",
            "filename": "glove-200-cosine.hdf5",
            "repo_type": "dataset",
            "revision": revision,
            "cache_dir": tmp_path / "cache",
        }
    ]


def test_hugging_face_source_can_feed_the_parquet_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    train_path = tmp_path / "cached-train.parquet"
    test_path = tmp_path / "cached-test.parquet"
    gt_path = tmp_path / "cached-neighbors.parquet"
    pl.DataFrame({"id": [0], "emb": [[1.0, 2.0]]}).write_parquet(train_path)
    pl.DataFrame({"id": [7], "emb": [[0.5, 0.25]]}).write_parquet(test_path)
    pl.DataFrame({"id": [7], "neighbors_id": [[0]]}).write_parquet(gt_path)

    data = CustomDataset(
        name="hf-parquet",
        size=1,
        dim=2,
        metric_type=MetricType.COSINE,
        use_shuffled=False,
        with_gt=True,
        with_remote_resource=True,
        dir="hf-parquet",
        file_num=1,
        source=DatasetSource.HuggingFace,
        source_dataset="VDBBench/example",
        source_revision="revision",
    )
    manager = ParquetDatasetManager(data=data)
    resolved = {
        "train.parquet": train_path,
        "test.parquet": test_path,
        "neighbors.parquet": gt_path,
    }

    class Reader:
        def read(self, *args, **kwargs):
            return resolved

    monkeypatch.setattr(DatasetSource, "reader", lambda _source: Reader())
    assert manager.prepare(k=1)
    assert manager.test_data == [[0.5, 0.25]]
    assert [row.tolist() for row in manager.gt_data.iter_rows()] == [[0]]
    [batch] = list(manager.iter_batches(1))
    assert batch["id"].tolist() == [0]


def _tiny_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distance: str,
) -> tuple[Hdf5DatasetManager, Path, np.ndarray]:
    metric = MetricType.L2 if distance == "euclidean" else MetricType.IP if distance == "ip" else MetricType.COSINE
    monkeypatch.setattr(config, "DATASET_LOCAL_DIR", tmp_path / "datasets")
    data = Hdf5Dataset(
        name=f"tiny-{distance}",
        size=5,
        dim=3,
        metric_type=metric,
        use_shuffled=False,
        with_gt=True,
        file_name=f"tiny-{distance}.hdf5",
        source=DatasetSource.HuggingFace,
        source_dataset="example/datasets",
        source_revision="test-revision",
        source_distance=distance,
        point_type="float",
        family="test",
        distribution="id",
        dataset_metadata={"distribution": "id"},
    )
    manager = Hdf5DatasetManager(data=data)
    source_path = tmp_path / data.file_name
    train = np.arange(15, dtype=np.float32).reshape(5, 3) / 7
    queries = np.array([[0.25, -0.5, 1.5], [3.25, 2.5, -1.0]], dtype=np.float32)
    neighbors = np.tile(np.arange(100, dtype=np.int64) % 5, (2, 1))
    with h5py.File(source_path, "w") as source:
        source.attrs["dimension"] = 3
        source.attrs["distance"] = distance
        source.attrs["point_type"] = "float"
        source.create_dataset("train", data=train)
        source.create_dataset("test", data=queries)
        source.create_dataset("neighbors", data=neighbors)
        source.create_dataset("distances", data=np.zeros((2, 100), dtype=np.float32))
    return manager, source_path, queries


@pytest.mark.parametrize("distance", ["euclidean", "normalized", "ip"])
def test_vibe_reads_hdf5_directly_with_one_open_per_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distance: str,
):
    manager, source_path, queries = _tiny_manager(tmp_path, monkeypatch, distance)
    calls = []

    class Reader:
        def read(
            self,
            dataset: str,
            files: list[str],
            local_ds_root: Path,
            *,
            revision: str | None = None,
        ) -> dict[str, Path]:
            calls.append((dataset, files, local_ds_root, revision))
            return {manager.data.file_name: source_path}

    monkeypatch.setattr(DatasetSource, "reader", lambda _source: Reader())
    assert manager.prepare(k=10)

    assert calls == [("example/datasets", [manager.data.file_name], manager.data_dir, "test-revision")]
    assert manager.test_data == queries.tolist()
    assert len(manager.gt_data) == len(queries)
    assert all(len(row) == 100 for row in manager.gt_data)
    assert manager.result_metadata["metric_type"] == manager.data.metric_type.value
    assert manager.result_metadata["revision"] == "test-revision"
    restored = pickle.loads(pickle.dumps(manager))
    assert restored.source_path == source_path

    real_hdf5_file = h5py.File
    load_opens = []

    def counting_hdf5_file(*args, **kwargs):
        load_opens.append(args[0])
        return real_hdf5_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", counting_hdf5_file)
    iterator = manager.iter_batches(batch_size=2)
    batches = list(iterator)
    assert load_opens == [source_path]
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [item for batch in batches for item in batch["id"].tolist()] == list(range(5))
    expected_train = np.arange(15, dtype=np.float32).reshape(5, 3) / np.float32(7)
    actual_train = np.concatenate([np.stack(batch["emb"]) for batch in batches])
    assert np.array_equal(actual_train, expected_train)
    assert iterator._file is None
    assert not list(tmp_path.rglob("*.parquet"))


def test_vibe_rejects_invalid_ground_truth_and_noncanonical_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager, source_path, _ = _tiny_manager(tmp_path, monkeypatch, "ip")
    with h5py.File(source_path, "r+") as source:
        source["neighbors"][0, 0] = 5

    class Reader:
        def read(self, *args, **kwargs):
            return {manager.data.file_name: source_path}

    monkeypatch.setattr(DatasetSource, "reader", lambda _source: Reader())
    with pytest.raises(ValueError, match="outside"):
        manager.prepare(k=10)

    assert manager.max_search_k(non_filter) == 100
    with pytest.raises(ValueError, match="K from 1 to 100"):
        manager.resolve_search_files(k=101)
    with pytest.raises(ValueError, match="does not contain scalar"):
        manager.resolve_search_files(k=10, filters=LabelFilter(label_percentage=0.5))


def test_vibe_case_cli_ui_and_preferred_source():
    case = Performance(dataset_name="glove-200-cosine")
    assert case.dataset.data.metric_type == MetricType.COSINE
    assert case.dataset.preferred_source == DatasetSource.HuggingFace
    assert isinstance(
        Performance(dataset_name=DatasetWithSizeType.CohereMedium.value).dataset,
        ParquetDatasetManager,
    )
    assert get_custom_case_config(
        {
            "case_type": "Performance",
            "dataset_name": "glove-200-cosine",
        }
    ) == {"dataset_name": "glove-200-cosine"}

    vibe = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "VIBE Search Performance")
    assert len(vibe.uiCaseItems) == 24

    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.Performance,
            custom_case={"dataset_name": "glove-200-cosine"},
        ),
    )
    runner = Assembler.assemble("run-id", task, DatasetSource.AliyunOSS)
    assert runner.dataset_source == DatasetSource.HuggingFace

    fts_case = CaseConfig(case_id=CaseType.FTSBm25Performance).case
    assert fts_case.dataset.preferred_source == DatasetSource.IR_DATASETS
    legacy_task = task.model_copy(update={"case_config": CaseConfig(case_id=CaseType.Performance768D1M)})
    assert Assembler.assemble("run-id", legacy_task, DatasetSource.AliyunOSS).dataset_source == DatasetSource.AliyunOSS

    with pytest.raises(ValueError, match="does not support filter"):
        Performance(dataset_name="glove-200-cosine", filter_rate=0.5)


@pytest.mark.parametrize(
    ("dataset_type", "legacy_case_type"),
    (
        (DatasetWithSizeType.CohereMedium, CaseType.Performance768D1M),
        (DatasetWithSizeType.CohereLarge, CaseType.Performance768D10M),
        (DatasetWithSizeType.LAIONLarge, CaseType.Performance768D100M),
        (DatasetWithSizeType.BioasqMedium, CaseType.Performance1024D1M),
        (DatasetWithSizeType.BioasqLarge, CaseType.Performance1024D10M),
        (DatasetWithSizeType.OpenAISmall, CaseType.Performance1536D50K),
        (DatasetWithSizeType.OpenAIMedium, CaseType.Performance1536D500K),
        (DatasetWithSizeType.OpenAILarge, CaseType.Performance1536D5M),
    ),
)
def test_registered_performance_preserves_legacy_timeouts(dataset_type, legacy_case_type):
    registered = Performance(dataset_name=dataset_type.value)
    legacy = type2case[legacy_case_type]()

    assert registered.load_timeout == legacy.load_timeout == registered.dataset.load_timeout
    assert registered.optimize_timeout == legacy.optimize_timeout == registered.dataset.optimize_timeout


@pytest.mark.parametrize(
    ("dataset_name", "load_timeout", "optimize_timeout"),
    (
        ("glove-200-cosine", config.LOAD_TIMEOUT_DEFAULT, config.OPTIMIZE_TIMEOUT_DEFAULT),
        ("dpr-jina-768-normalized", config.LOAD_TIMEOUT_768D_10M, config.OPTIMIZE_TIMEOUT_768D_10M),
        ("msmarco-qwen-1024-normalized", config.LOAD_TIMEOUT_1024D_10M, config.OPTIMIZE_TIMEOUT_1024D_10M),
        ("hotpotqa-harrier-640-normalized", config.LOAD_TIMEOUT_768D_10M, config.OPTIMIZE_TIMEOUT_768D_10M),
        ("multimodal-embedding-10m", config.LOAD_TIMEOUT_768D_10M, config.OPTIMIZE_TIMEOUT_768D_10M),
        ("multimodal-embedding-100m", config.LOAD_TIMEOUT_768D_100M, config.OPTIMIZE_TIMEOUT_768D_100M),
    ),
)
def test_new_registered_datasets_declare_performance_timeouts(dataset_name, load_timeout, optimize_timeout):
    case = Performance(dataset_name=dataset_name)

    assert case.load_timeout == case.dataset.load_timeout == load_timeout
    assert case.optimize_timeout == case.dataset.optimize_timeout == optimize_timeout


def test_registered_dataset_sizes_remain_separate_in_frontend_results():
    dataset_names = (DatasetWithSizeType.CohereSmall.value, DatasetWithSizeType.CohereLarge.value)
    results = [
        CaseResult(
            metrics=Metric(),
            task_config=TaskConfig(
                db=DB.Test,
                db_config=DB.Test.config_cls(),
                db_case_config=EmptyDBCaseConfig(),
                case_config=CaseConfig(case_id=CaseType.Performance, custom_case={"dataset_name": dataset_name}),
            ),
        )
        for dataset_name in dataset_names
    ]

    merged, failed = mergeTasks(results)

    assert not failed
    assert {result["case_name"] for result in merged} == {
        f"Search Performance - {dataset_name}" for dataset_name in dataset_names
    }


def test_vibe_k_above_100_fails_during_case_config_validation():
    with pytest.raises(ValueError, match="K from 1 to 100"):
        CaseConfig(
            case_id=CaseType.Performance,
            custom_case={"dataset_name": "glove-200-cosine"},
            k=101,
        )


def test_vibe_result_metadata_is_optional_and_round_trips(tmp_path: Path):
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.Performance,
            custom_case={"dataset_name": "glove-200-cosine"},
        ),
    )
    old_result = TestResult(
        run_id="old-vibe",
        task_label="old-vibe",
        results=[CaseResult(metrics=Metric(), task_config=task)],
    )
    old_payload = old_result.model_dump_for_output()
    old_payload["results"][0].pop("dataset_metadata")
    old_path = tmp_path / "old-result.json"
    old_path.write_text(json.dumps(old_payload), encoding="utf-8")
    assert TestResult.read_file(old_path).results[0].dataset_metadata is None

    metadata = {
        "name": "glove-200-cosine",
        "distribution": "id",
        "source": "HuggingFace",
        "repository": "vector-index-bench/vibe",
        "filename": "glove-200-cosine.hdf5",
        "revision": "07b387891a221b7b073b83d2f752b76462e5fa03",
        "source_distance": "cosine",
        "metric_type": "COSINE",
        "point_type": "float",
    }
    case_result = CaseResult(metrics=Metric(), task_config=task, dataset_metadata=metadata)
    payload = case_result.model_dump(mode="json")
    assert json.loads(json.dumps(payload))["dataset_metadata"] == metadata

    test_result = TestResult(run_id="vibe", task_label="vibe", results=[case_result])
    result_path = tmp_path / "vibe-result.json"
    result_path.write_text(json.dumps(test_result.model_dump_for_output()), encoding="utf-8")
    loaded_metadata = TestResult.read_file(result_path).results[0].dataset_metadata
    assert loaded_metadata is not None
    assert loaded_metadata.model_dump(mode="json") == metadata
    [rest_payload] = format_results([test_result], "vibe")
    assert rest_payload["dataset_metadata"] == metadata
    merged, failed = mergeTasks([case_result])
    assert not failed
    assert merged[0]["dataset_metadata"] == metadata
