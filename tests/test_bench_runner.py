import time
import logging
from vector_db_bench.interface import BenchMarkRunner
from vector_db_bench.models import (
    DB,
    IndexType, CaseType,
    TaskConfig, CaseConfig,
)

log = logging.getLogger(__name__)

class TestBenchRunner:
    def test_performance_case_whole(self):
        runner = BenchMarkRunner()

        task_config=TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        runner._sync_running_task()
        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_clean(self):
        runner = BenchMarkRunner()

        task_config=TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        time.sleep(3)
        runner.stop_running()
