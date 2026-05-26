import json
import logging

import pytest
from vectordb_bench.models import (
    TaskConfig, CaseConfig,
    CaseResult, TestResult,
    Metric, CaseType
)
from vectordb_bench.backend.clients import (
    DB,
    IndexType
)

from vectordb_bench import config


log = logging.getLogger("vectordb_bench")


def test_result_read_file_allows_display_with_sanitized_db_config(tmp_path):
    json_file = tmp_path / "result_sanitized_pinecone.json"
    json_file.write_text(
        json.dumps(
            {
                "run_id": "sanitized-config",
                "task_label": "standard",
                "results": [
                    {
                        "metrics": {
                            "max_load_count": 1000,
                            "load_duration": 0,
                            "qps": 0,
                            "serial_latency_p99": 0.1,
                            "recall": 0,
                        },
                        "task_config": {
                            "db": "Pinecone",
                            "db_config": {
                                "db_label": "pinecone-serverless",
                                "version": "",
                                "note": "",
                            },
                            "db_case_config": {
                                "null": None,
                            },
                            "case_config": {
                                "case_id": CaseType.Performance768D1M.value,
                                "custom_case": {},
                            },
                        },
                        "label": ":)",
                    }
                ],
            }
        )
    )

    result = TestResult.read_file(json_file, trans_unit=True)

    assert result.results[0].task_config.db_config.db_label == "pinecone-serverless"
    assert result.results[0].metrics.max_load_count == 1


class TestModels:
    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.Milvus,
                db_config=DB.Milvus.config(),
                db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
                case_config=CaseConfig(case_id=CaseType.Performance10M),
            ),
            metrics=Metric(),
        )

        test_result = TestResult(run_id=10000, results=[result])
        test_result.flush()

        with pytest.raises(ValueError):
            result = TestResult.read_file('nosuchfile.json')

    def test_test_result_read_write(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            res = TestResult.read_file(json_file)
            res.flush()

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
        tr.flush()

    def test_test_result_display(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            log.info(json_file)
            res = TestResult.read_file(json_file)
            res.display()
