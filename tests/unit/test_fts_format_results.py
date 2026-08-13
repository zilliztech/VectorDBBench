from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.milvus.config import MilvusConfig, MilvusFtsConfig
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig, TestResult
from vectordb_bench.restful.format_res import format_results


def test_format_results_supports_fts_dataset_and_metrics():
    test_result = TestResult(
        run_id="run-1",
        task_label="fts-task",
        timestamp=123,
        results=[
            CaseResult(
                task_config=TaskConfig(
                    db=DB.Milvus,
                    db_config=MilvusConfig(uri="http://localhost:19530"),
                    db_case_config=MilvusFtsConfig(),
                    case_config=CaseConfig(case_id=CaseType.FTSBm25Performance, k=10),
                ),
                metrics=Metric(
                    qps=12.5,
                    recall=0.7,
                    ndcg=0.8,
                    mrr=0.9,
                    serial_latency_p99=0.11,
                    serial_latency_p95=0.1,
                    conc_latency_p99_list=[0.2],
                    conc_latency_p95_list=[0.15],
                    conc_latency_avg_list=[0.12],
                ),
            )
        ],
    )

    [formatted] = format_results([test_result], "fts-task")

    assert formatted["dataset"] == "MS MARCO FTS (SMALL)"
    assert formatted["dim"] == 0
    assert formatted["mrr"] == 0.9
    assert formatted["serial_latency_p95"] == 0.1
    assert formatted["conc_latency_p95_list"] == [0.15]
