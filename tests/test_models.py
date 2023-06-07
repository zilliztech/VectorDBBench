import pytest
import logging
from vector_db_bench.models import (
    DB,
    IndexType, MetricType, CaseType,
    TaskConfig, CaseConfig,
    CaseResult, TestResult,
    Metric,
)

from vector_db_bench.backend.clients.milvus.milvus.config import (
    MilvusConfig, HNSWConfig, IVFFlatConfig,
)
from vector_db_bench import config


log = logging.getLogger(__name__)


class TestModels:
    def test_db_milvus(self):
        assert DB.Milvus.value == "Milvus"
        assert DB.Milvus.config == MilvusConfig
        assert DB.Milvus.case_config_cls(IndexType.HNSW) == HNSWConfig
        assert DB.Milvus.case_config_cls(IndexType.IVFFlat) == IVFFlatConfig

        milvus_case_config_cls = DB.Milvus.case_config_cls(IndexType.Flat)
        c = milvus_case_config_cls(metric_type=MetricType.COSINE)
        assert c.index_param() == {
            'metric_type': "L2",
            'index_type': "FLAT",
            'params': {},
        }

        assert c.search_param() == {
            "metric_type": "L2",
            "params": {},
        }

        c = milvus_case_config_cls()
        c.metric_type = MetricType.COSINE
        assert c.index_param() == {
            'metric_type': "L2",
            'index_type': "FLAT",
            'params': {},
        }

        assert c.search_param() == {
            "metric_type": "L2",
            "params": {},
        }

        with pytest.raises(AssertionError):
            DB.Milvus.case_config_cls()

    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            result_id=100,
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
