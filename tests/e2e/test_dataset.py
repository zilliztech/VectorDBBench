import logging
import os
import time

import pytest

from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import Dataset

log = logging.getLogger("vectordb_bench")
pytestmark = pytest.mark.integration


class TestDataSet:
    def test_iter_cohere(self):
        cohere_10m = Dataset.COHERE.manager(10_000_000)
        cohere_10m.prepare()

        before = time.time()
        for batch in cohere_10m:
            log.debug(batch.head(1))

        duration = time.time() - before
        log.warning("iter through cohere_10m cost=%smin", duration / 60)

    def test_iter_laion(self):
        laion_100m = Dataset.LAION.manager(100_000_000)
        laion_100m.prepare(source=DatasetSource.AliyunOSS)

        before = time.time()
        for batch in laion_100m:
            log.debug(batch.head(1))

        duration = time.time() - before
        log.warning("iter through laion_100m cost=%smin", duration / 60)

    def test_download_small(self):
        openai_50k = Dataset.OPENAI.manager(50_000)
        files = [
            "test.parquet",
            "neighbors.parquet",
            "neighbors_head_1p.parquet",
            "neighbors_tail_1p.parquet",
        ]

        file_path = openai_50k.data_dir.joinpath("test.parquet")
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
