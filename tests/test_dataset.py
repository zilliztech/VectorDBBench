import pytest
import logging

from falcon_mark.backend import dataset as ds

log = logging.getLogger(__name__)
class TestDataSet:
    def test_init_dataset(self):
        g = ds.GIST_S()
        d = ds.DataSetManager(data=g, data_path="/path")
        log.debug(f"dataset: {d.dict()}\n gist: {d.data}\n label: {d.data.label}")

    def test_init_gist(self):
        g = ds.GIST_S()
        log.debug(f"GIST SMALL: {g}")
        assert g.name == "GIST_S_100K"
        assert g.label == "SMALL"
        assert g.size == 100_000

    def test_get_dataset(self):
        testcases = [
            (ds.DataSet.GIST, ds.Label.SMALL),
            (ds.DataSet.GIST, ds.Label.MEDIUM),
        ]

        for t in testcases:
            dataset = ds.get_data_set(*t)
            assert dataset.label == t[1].name

