import logging
import vector_db_bench.backend.dataset as ds
from vector_db_bench.models import DB
from vector_db_bench.backend import cases
from vector_db_bench.backend.clients.milvus import Milvus, FLATConfig

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
        db_config = DB.Milvus.config().to_dict()
        db_case_config = FLATConfig(metric_type=dataset.data.metric_type)
        milvus = Milvus(
            db_config=db_config,
            db_case_config=db_case_config,
            drop_old=True,
        )

        c = cases.PerformanceSZero(run_id=1, db=milvus)

        #  c.dataset.prepare()
        #  c.search()

        c.run()
