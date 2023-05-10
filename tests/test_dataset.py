import pytest
import logging

from falcon_mark.backend import dataset as ds

log = logging.getLogger(__name__)
class TestDataSet:
    def test_init_dataset(self):
        log.debug("a")
        testdatasets = [ds.get(d, lb) for d in ds.Name for lb in ds.Label if ds.get(d, lb) is not None]
        for t in testdatasets:
            log.debug(f"dir name: {t.data.dir_name}, dataset: {t.model_dump()}")

    def test_init_gist(self):
        g = ds.GIST_S()
        log.debug(f"GIST SMALL: {g}")
        assert g.name == "GIST"
        assert g.label == "SMALL"
        assert g.size == 100_000
