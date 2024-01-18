from vectordb_bench.backend.dataset import Dataset, get_files
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


class TestGetFiles:
    @pytest.mark.parametrize("train_count", [
        1,
        10,
        50,
        100,
    ])
    @pytest.mark.parametrize("with_gt", [True, False])
    def test_train_count(self, train_count, with_gt):
        files = get_files(train_count, True, with_gt)
        log.info(files)

        if with_gt:
            assert len(files) - 4 == train_count
        else:
            assert len(files) - 1 == train_count

    @pytest.mark.parametrize("use_shuffled", [True, False])
    def test_use_shuffled(self, use_shuffled):
        files = get_files(1, use_shuffled, True)
        log.info(files)

        trains = [f for f in files if "train" in f]
        if use_shuffled:
            for t in trains:
                assert "shuffle_train" in t
        else:
            for t in trains:
                assert "shuffle" not in t
                assert "train" in t
