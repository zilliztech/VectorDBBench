import logging
import pathlib
import pytest
from vectordb_bench.backend.data_source import AliyunOSSReader, AwsS3Reader
from vectordb_bench.backend.dataset import Dataset, DatasetManager

log = logging.getLogger(__name__)

class TestReader:
    @pytest.mark.parametrize("size", [
        100_000,
        1_000_000,
        10_000_000,
    ])
    def test_cohere(self, size):
        cohere = Dataset.COHERE.manager(size)
        self.per_dataset_test(cohere)

    @pytest.mark.parametrize("size", [
        100_000,
        1_000_000,
    ])
    def test_gist(self, size):
        gist = Dataset.GIST.manager(size)
        self.per_dataset_test(gist)

    @pytest.mark.parametrize("size", [
        1_000_000,
    ])
    def test_glove(self, size):
        glove = Dataset.GLOVE.manager(size)
        self.per_dataset_test(glove)

    @pytest.mark.parametrize("size", [
        500_000,
        5_000_000,
        #  50_000_000,
    ])
    def test_sift(self, size):
        sift = Dataset.SIFT.manager(size)
        self.per_dataset_test(sift)

    @pytest.mark.parametrize("size", [
        50_000,
        500_000,
        5_000_000,
    ])
    def test_openai(self, size):
        openai = Dataset.OPENAI.manager(size)
        self.per_dataset_test(openai)


    def per_dataset_test(self, dataset: DatasetManager):
        s3_reader = AwsS3Reader()
        all_files = s3_reader.ls_all(dataset.data.dir_name)


        remote_f_names = []
        for file in all_files:
            remote_f = pathlib.Path(file).name
            if dataset.data.use_shuffled and remote_f.startswith("train"):
                continue

            elif (not dataset.data.use_shuffled) and remote_f.startswith("shuffle"):
                continue

            remote_f_names.append(remote_f)


        assert set(dataset.data.files) == set(remote_f_names)

        aliyun_reader = AliyunOSSReader()
        for fname in dataset.data.files:
            p = pathlib.Path("benchmark", dataset.data.dir_name, fname)
            assert aliyun_reader.bucket.object_exists(p.as_posix())

        log.info(f"downloading to {dataset.data_dir}")
        aliyun_reader.read(dataset.data.dir_name.lower(), dataset.data.files, dataset.data_dir)
