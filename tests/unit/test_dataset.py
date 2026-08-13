import logging

import pytest
from pydantic import ValidationError

from vectordb_bench.backend.dataset import Dataset


log = logging.getLogger("vectordb_bench")


class TestDataSet:
    def test_iter_dataset(self):
        for dataset in Dataset:
            log.info(dataset)

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
