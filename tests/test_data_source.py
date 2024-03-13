import logging
import pytest
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.cases import type2case

log = logging.getLogger("vectordb_bench")

class TestReader:
    @pytest.mark.parametrize("type_case", [
        (k, v) for k, v in type2case.items()
    ])
    def test_type_cases(self, type_case):
        self.per_case_test(type_case)


    def per_case_test(self, type_case):
        t, ca_cls = type_case
        ca = ca_cls()
        log.info(f"test case: {t.name}, {ca.name}")

        filters = ca.filter_rate
        ca.dataset.prepare(source=DatasetSource.AliyunOSS, filters=filters)
        ali_trains = ca.dataset.train_files

        ca.dataset.prepare(filters=filters)
        s3_trains = ca.dataset.train_files

        assert ali_trains == s3_trains
