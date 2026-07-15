import struct

from vectordb_bench import config
from vectordb_bench.backend.clients import MetricType
from vectordb_bench.backend.dataset import BinaryDatasetManager, Dataset, DatasetWithSizeType, SIFTBinary
import logging
import pytest
from pydantic import ValidationError
from vectordb_bench.backend.data_source import DatasetSource

log = logging.getLogger("vectordb_bench")


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

    def test_sift_binary_dataset_type(self):
        dataset = DatasetWithSizeType.SIFTBinary1M.get_manager()

        assert isinstance(dataset, BinaryDatasetManager)
        assert dataset.data.name == "SIFTBinary"
        assert dataset.data.dim == 128
        assert dataset.data.metric_type == MetricType.HAMMING

    def test_binary_dataset_manager_reads_vec_tool_bin_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATASET_LOCAL_DIR", tmp_path)
        vectors = [
            bytes(range(16)),
            bytes([255] * 16),
            bytes([170] * 16),
        ]
        truth = [[2, 1], [0, 2]]

        data = SIFTBinary(size=1_000_000)
        manager = BinaryDatasetManager(data=data)
        manager.data_dir.mkdir(parents=True)
        manager.data_dir.joinpath("base.bin").write_bytes(struct.pack("<II", len(vectors), 128) + b"".join(vectors))
        manager.data_dir.joinpath("query.bin").write_bytes(struct.pack("<II", 1, 128) + vectors[0])
        manager.data_dir.joinpath("truth.ibin").write_bytes(
            struct.pack("<II", len(truth), len(truth[0])) + b"".join(struct.pack("<i", v) for row in truth for v in row)
        )

        assert manager._read_binary_vectors("query.bin") == [vectors[0].hex()]
        assert manager._read_truth_ids("truth.ibin") == truth

        batches = list(manager.iter_batches(batch_size=2))
        assert batches[0]["id"].tolist() == [0, 1]
        assert batches[0]["emb"].tolist() == [vectors[0].hex(), vectors[1].hex()]
        assert batches[1]["id"].tolist() == [2]
        assert batches[1]["emb"].tolist() == [vectors[2].hex()]
