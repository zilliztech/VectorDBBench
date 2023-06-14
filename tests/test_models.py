import pytest
import logging
from vector_db_bench.models import (
    TaskConfig, CaseConfig,
    CaseResult, TestResult,
    Metric, CaseType
)
from vector_db_bench.backend.clients import (
    DB,
    IndexType
)

from vector_db_bench import config


log = logging.getLogger("vector_db_bench")


class TestModels:
    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.Milvus,
                db_config=DB.Milvus.config(),
                db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
                case_config=CaseConfig(case_id=CaseType.PerformanceLZero),
            ),
            metrics=Metric(),
        )

        test_result = TestResult(run_id=10000, results=[result])
        test_result.write_file()

        with pytest.raises(ValueError):
            result = TestResult.read_file('nosuchfile.json')

    def test_test_result_read_write(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)
            res.task_label = f"Milvus-{res.run_id}"
            res.write_file()

    def test_test_result_merge(self):
        result_dir = config.RESULTS_LOCAL_DIR
        all_results = []

        first_result = None
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)

            for cr in res.results:
                all_results.append(cr)

            if not first_result:
                first_result = res

        tr = TestResult(
            run_id=first_result.run_id,
            task_label="standard",
            results=all_results,
        )
        tr.write_file()

    def test_test_result_display(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)
            res.display()
