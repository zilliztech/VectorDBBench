import logging
import pickle

import polars as pl
import pytest
from pydantic import ValidationError

from vectordb_bench import config
from vectordb_bench.backend import dataset as dataset_module
from vectordb_bench.backend.clients import MetricType
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import CustomDataset, Dataset, DatasetManager
from vectordb_bench.backend.filter import LabelFilter, NewIntFilter, non_filter

log = logging.getLogger("vectordb_bench")


@pytest.mark.parametrize(
    ("k", "test_file", "gt_file", "width", "query_count"),
    [
        (1_000, "test.parquet", "neighbors.parquet", 1_000, 1_000),
        (1_001, "test_nq200.parquet", "neighbors_top100k_nq200.parquet", 100_000, 200),
        (100_000, "test_nq200.parquet", "neighbors_top100k_nq200.parquet", 100_000, 200),
        (100_001, "test_nq200.parquet", "neighbors_top1m_nq200.parquet", 1_000_000, 200),
        (1_000_000, "test_nq200.parquet", "neighbors_top1m_nq200.parquet", 1_000_000, 200),
    ],
)
def test_laion_artifact_selection(k, test_file, gt_file, width, query_count):
    dataset = Dataset.LAION.manager(100_000_000)
    assert hasattr(dataset, "resolve_search_files")

    files = dataset.resolve_search_files(k=k, filters=non_filter)

    assert files.test_file == test_file
    assert files.gt_file == gt_file
    assert files.width == width
    assert files.query_count == query_count


@pytest.mark.parametrize("k", [0, -1, 1_000_001])
def test_laion_artifact_selection_rejects_unsupported_k(k):
    dataset = Dataset.LAION.manager(100_000_000)
    assert hasattr(dataset, "resolve_search_files")

    with pytest.raises(ValueError, match="LAION"):
        dataset.resolve_search_files(k=k, filters=non_filter)


def _laion_int_filter(filter_rate: float) -> NewIntFilter:
    return NewIntFilter(filter_rate=filter_rate, int_field="id", int_value=int(100_000_000 * filter_rate))


def test_laion_integer_filter_keeps_standard_artifacts_at_k_1000():
    dataset = Dataset.LAION.manager(100_000_000)
    files = dataset.resolve_search_files(k=1_000, filters=_laion_int_filter(0.5))

    assert files.test_file == "test.parquet"
    assert files.gt_file == "neighbors_int_50p.parquet"


@pytest.mark.parametrize(
    ("filter_rate", "max_k"),
    [
        (0.5, 1_000_000),
        (0.6, 1_000_000),
        (0.7, 1_000_000),
        (0.8, 1_000_000),
        (0.9, 1_000_000),
        (0.95, 1_000_000),
        (0.98, 1_000_000),
        (0.99, 1_000_000),
        (0.995, 500_000),
        (0.998, 200_000),
        (0.999, 100_000),
    ],
)
def test_laion_integer_filter_selects_published_maximum_gt(filter_rate, max_k):
    dataset = Dataset.LAION.manager(100_000_000)
    filters = _laion_int_filter(filter_rate)
    files = dataset.resolve_search_files(k=max_k, filters=filters)
    width_suffix = f"{max_k // 1_000_000}m" if max_k >= 1_000_000 else f"{max_k // 1_000}k"

    assert files.test_file == "test_nq200.parquet"
    assert files.gt_file == f"neighbors_{filters.int_rate}_top{width_suffix}_nq200.parquet"
    assert files.width == max_k
    assert files.query_count == 200


def test_laion_integer_filter_selects_top100k_for_smaller_large_topk():
    dataset = Dataset.LAION.manager(100_000_000)
    files = dataset.resolve_search_files(k=1_001, filters=_laion_int_filter(0.995))

    assert files.gt_file == "neighbors_int_99.5p_top100k_nq200.parquet"
    assert files.width == 100_000


@pytest.mark.parametrize(
    ("filters", "k", "error"),
    [
        (_laion_int_filter(0.75), 1_000, "supported filter rates"),
        (_laion_int_filter(0.999), 100_001, "supports K up to 100,000"),
        (LabelFilter(label_percentage=0.5), 1_001, "integer filters"),
    ],
)
def test_laion_filtered_artifact_selection_rejects_unpublished_combinations(filters, k, error):
    dataset = Dataset.LAION.manager(100_000_000)

    with pytest.raises(ValueError, match=error):
        dataset.resolve_search_files(k=k, filters=filters)


def test_dataset_prepare_keeps_ground_truth_path_based(tmp_path, monkeypatch):
    assert hasattr(dataset_module, "ParquetGroundTruth")
    monkeypatch.setattr(config, "DATASET_LOCAL_DIR", tmp_path)
    dataset = _custom_dataset_manager()
    dataset.data.with_remote_resource = False
    dataset.data_dir.mkdir(parents=True)
    _write_vector_fixture(dataset.data_dir)

    dataset.prepare(with_train_files=False, k=4)

    assert isinstance(dataset.gt_data, dataset_module.ParquetGroundTruth)
    assert dataset.gt_data.row_count == 2
    assert dataset.gt_data.width == 4
    restored = pickle.loads(pickle.dumps(dataset.gt_data))
    assert [row.tolist() for row in restored.iter_rows()] == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_parquet_ground_truth_rejects_query_id_mismatch(tmp_path):
    assert hasattr(dataset_module, "ParquetGroundTruth")
    gt_path = tmp_path / "neighbors.parquet"
    pl.DataFrame({"id": [11, 20], "neighbors_id": [[1, 2], [3, 4]]}).write_parquet(gt_path)

    with pytest.raises(ValueError, match="query IDs"):
        dataset_module.ParquetGroundTruth.from_file(
            gt_path,
            id_field="id",
            neighbors_field="neighbors_id",
            expected_query_ids=[10, 20],
            minimum_width=2,
        )


def test_parquet_ground_truth_rejects_narrow_row(tmp_path):
    assert hasattr(dataset_module, "ParquetGroundTruth")
    gt_path = tmp_path / "neighbors.parquet"
    pl.DataFrame({"id": [10, 20], "neighbors_id": [[1, 2], [3]]}).write_parquet(gt_path)

    with pytest.raises(ValueError, match="width"):
        dataset_module.ParquetGroundTruth.from_file(
            gt_path,
            id_field="id",
            neighbors_field="neighbors_id",
            expected_query_ids=[10, 20],
            minimum_width=2,
        )


def _custom_dataset_manager() -> DatasetManager:
    data = CustomDataset(
        name="local",
        size=8,
        dim=2,
        metric_type=MetricType.L2,
        use_shuffled=False,
        with_gt=True,
        dir="large_topk_fixture",
        file_num=1,
    )
    return DatasetManager(data=data)


def _write_vector_fixture(data_dir):
    pl.DataFrame(
        {
            "id": [10, 20],
            "emb": [[0.1, 0.2], [0.3, 0.4]],
        }
    ).write_parquet(data_dir / "test.parquet")
    pl.DataFrame(
        {
            "id": [10, 20],
            "neighbors_id": [[1, 2, 3, 4], [5, 6, 7, 8]],
        }
    ).write_parquet(data_dir / "neighbors.parquet")


class TestDataSet:
    def test_iter_dataset(self):
        for ds in Dataset:
            log.info(ds)

    def test_cohere(self):
        cohere = Dataset.COHERE.get(100_000)
        log.info(cohere)
        assert cohere.name == "Cohere"
        assert cohere.size == 100_000
        assert cohere.label == "SMALL"
        assert cohere.dim == 768

    def test_cohere_error(self):
        with pytest.raises(ValidationError):
            Dataset.COHERE.get(9999)

    def test_iter_cohere(self):
        cohere_10m = Dataset.COHERE.manager(10_000_000)
        cohere_10m.prepare()

        import time

        before = time.time()
        for i in cohere_10m:
            log.debug(i.head(1))

        dur_iter = time.time() - before
        log.warning(f"iter through cohere_10m cost={dur_iter/60}min")

    # pytest -sv tests/test_dataset.py::TestDataSet::test_iter_laion
    def test_iter_laion(self):
        laion_100m = Dataset.LAION.manager(100_000_000)
        from vectordb_bench.backend.data_source import DatasetSource

        laion_100m.prepare(source=DatasetSource.AliyunOSS)

        import time

        before = time.time()
        for i in laion_100m:
            log.debug(i.head(1))

        dur_iter = time.time() - before
        log.warning(f"iter through laion_100m cost={dur_iter/60}min")

    def test_download_small(self):
        openai_50k = Dataset.OPENAI.manager(50_000)
        files = [
            "test.parquet",
            "neighbors.parquet",
            "neighbors_head_1p.parquet",
            "neighbors_tail_1p.parquet",
        ]

        file_path = openai_50k.data_dir.joinpath("test.parquet")
        import os

        DatasetSource.S3.reader().read(
            openai_50k.data.dir_name.lower(),
            files=files,
            local_ds_root=openai_50k.data_dir,
        )

        os.remove(file_path)
        DatasetSource.AliyunOSS.reader().read(
            openai_50k.data.dir_name.lower(),
            files=files,
            local_ds_root=openai_50k.data_dir,
        )
