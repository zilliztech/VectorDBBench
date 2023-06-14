import pytest
import logging

from vectordb_bench.backend import dataset as ds

log = logging.getLogger(__name__)
class TestDataSet:
    @pytest.mark.skip("not ready in s3")
    def test_init_dataset(self):
        testdatasets = [ds.get(d, lb) for d in ds.Name for lb in ds.Label if ds.get(d, lb) is not None]
        for t in testdatasets:
            t._validate_local_file()

    @pytest.mark.skip("not ready in s3")
    def test_init_gist(self):
        g = ds.GIST_S()
        log.debug(f"GIST SMALL: {g}")
        assert g.name == "GIST"
        assert g.label == "SMALL"
        assert g.size == 100_000

        gists = [ds.get(ds.Name.GIST, lb) for lb in ds.Label if ds.get(ds.Name.GIST, lb) is not None]
        for t in gists:
            t._validate_local_file()

    def test_init_cohere(self):
        coheres = [ds.get(ds.Name.Cohere, lb) for lb in ds.Label if ds.get(ds.Name.Cohere, lb) is not None]
        for t in coheres:
            t._validate_local_file()

    def test_init_sift(self):
        sifts = [ds.get(ds.Name.SIFT, lb) for lb in ds.Label if ds.get(ds.Name.SIFT, lb) is not None]
        for t in sifts:
            t._validate_local_file()

    @pytest.mark.skip("runs locally")
    def test_iter_dataset_cohere(self):
        cohere_s = ds.get(ds.Name.Cohere, ds.Label.SMALL)
        assert cohere_s.prepare()

        for f in cohere_s:
            log.debug(f"iter to: {f.columns}")

    #  @pytest.mark.skip("runs locally")
    def test_dataset_download(self):
        cohere_s = ds.get(ds.Name.Cohere, ds.Label.SMALL)
        assert cohere_s.prepare()


        cohere_m = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
        cohere_m._validate_local_file()
        assert cohere_m.prepare() is True
        assert cohere_m.prepare() is True
