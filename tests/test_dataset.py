from vectordb_bench.backend.dataset import Dataset
import logging
import pytest
from pydantic import ValidationError


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

    def test_init_cohere(self):
        coheres = [Dataset.COHERE.manager(i) for i in [100_000, 1_000_000, 10_000_000]]
        for t in coheres:
            t._validate_local_file()

    def test_iter_cohere(self):
        cohere_10m = Dataset.COHERE.manager(10_000_000)
        cohere_10m.prepare(False)
        for i in cohere_10m:
            log.debug(i.head(1))

