import pytest
import logging
import vector_db_bench.backend.dataset as ds
from vector_db_bench.models import DB, IndexType
from vector_db_bench.backend import cases
from vector_db_bench.backend.clients.milvus import Milvus
from vector_db_bench.backend.clients.weaviate import Weaviate

log  = logging.getLogger(__name__)
class TestCases:
    def test_init_LoadCase(self):
        c = cases.LoadSDimCase(run_id=1, db_class=Milvus)
        log.debug(f"c: {c}, {c.model_dump().keys()}")

    def test_case_type(self):
        from vector_db_bench.models import CaseType
        log.debug(f"{CaseType.LoadLDim}")

    def test_performance_case_small_zero(self):
        dataset = ds.get(ds.Name.Cohere, ds.Label.SMALL)
        # milvus crash
        #  db_case_config = DB.Milvus.case_config_cls(IndexType.HNSW)(
        #      M=8,
        #      efConstruction=32,
        #      ef=8,
        #  )

        db_case_config = DB.Milvus.case_config_cls(IndexType.Flat)()
        db_case_config.metric_type = dataset.data.metric_type
        c = cases.PerformanceSZero(run_id=1, db_configs=(
            DB.Milvus.init_cls,
            DB.Milvus.config().to_dict(),
            db_case_config,
        ))
        c.run()

    @pytest.mark.skip(reason="replace url and auth_key by real value")
    def test_performance_case_small_zero_weaviate(self):
        dataset = ds.get(ds.Name.Cohere, ds.Label.SMALL)
        db_case_config = DB.Weaviate.case_config_cls()()
        db_case_config.metric_type = dataset.data.metric_type

        c = cases.PerformanceSZero(run_id=1, db_configs={
            DB.Weaviate.init_cls,
            DB.Weaviate.config(url="", auth_key="").to_dict(),
            db_case_config,
        })
        c.run()

    def test_performance_case_small_low_filter(self):
        dataset = ds.get(ds.Name.Cohere, ds.Label.SMALL)

        db_case_config = DB.Milvus.case_config_cls(IndexType.Flat)()
        db_case_config.metric_type = dataset.data.metric_type
        c = cases.PerformanceSLow(run_id=2, db_configs=(
            DB.Milvus.init_cls,
            DB.Milvus.config().to_dict(),
            db_case_config,
        ))
        c.run()

    def test_performance_case_small_high_filter(self):
        dataset = ds.get(ds.Name.Cohere, ds.Label.SMALL)
        db_case_config = DB.Milvus.case_config_cls(IndexType.Flat)()
        db_case_config.metric_type = dataset.data.metric_type

        c = cases.PerformanceSHigh(run_id=3, db_configs=(
            DB.Milvus.init_cls,
            DB.Milvus.config().to_dict(),
            db_case_config,
        ))
        c.run()

    #  @pytest.mark.skip("wait for sift in s3")
    def test_load_small_dim(self):
        dataset = ds.get(ds.Name.SIFT, ds.Label.SMALL)
        db_case_config = DB.Milvus.case_config_cls(IndexType.Flat)()
        db_case_config.metric_type = dataset.data.metric_type

        c = cases.LoadSDimCase(run_id=1, db_configs=(
            DB.Milvus.init_cls,
            DB.Milvus.config().to_dict(),
            db_case_config,
        ))
        c.run()

