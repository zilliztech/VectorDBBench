import pytest
import logging
from vector_db_bench.backend import cases
from vector_db_bench.backend.clients.milvus import Milvus

log  = logging.getLogger(__name__)
class TestCases:
    def test_init_LoadCase(self):
        c = cases.LoadSDimCase(run_id=1, db_class=Milvus)
        log.debug(f"c: {c}, {c.model_dump().keys()}")

    def test_case_type(self):
        from vector_db_bench.models import CaseType
        log.debug(f"{CaseType.LoadLDim}")

    def test_performance_case_small_zero(self):
        c = cases.PerformanceSZero(run_id=1, db_class=Milvus)
        c.run()
