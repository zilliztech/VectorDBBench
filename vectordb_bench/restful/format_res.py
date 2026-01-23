from dataclasses import asdict

from pydantic import BaseModel

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.models import TestResult


class FormatResult(BaseModel):
    # db_config
    task_label: str = ""
    timestamp: int = 0
    db: str = ""
    db_label: str = ""  # perf-x86
    version: str = ""
    note: str = ""

    # params
    params: dict = {}

    # case_config
    case_name: str = ""
    dataset: str = ""
    dim: int = 0
    filter_type: str = ""  # FilterType(Enum).value
    filter_rate: float = 0
    k: int = 100

    # metrics
    max_load_count: int = 0
    load_duration: int = 0
    qps: float = 0
    serial_latency_p99: float = 0
    recall: float = 0
    ndcg: float = 0
    conc_num_list: list[int] = []
    conc_qps_list: list[float] = []
    conc_latency_p99_list: list[float] = []
    conc_latency_avg_list: list[float] = []


def format_results(test_results: list[TestResult], task_label: str) -> list[dict]:
    results = []
    for test_result in test_results:
        if test_result.task_label == task_label:
            for case_result in test_result.results:
                task_config = case_result.task_config
                case_config = task_config.case_config
                case = case_config.case
                if case.label == CaseLabel.Load:
                    continue
                dataset = case.dataset.data
                filter_ = case.filters
                metrics = asdict(case_result.metrics)
                for k, v in metrics.items():
                    if isinstance(v, list) and len(v) > 0:
                        metrics[k] = [round(d, 6) if isinstance(d, float) else d for d in v]
                results.append(
                    FormatResult(
                        task_label=test_result.task_label,
                        timestamp=int(test_result.timestamp),
                        db=task_config.db.value,
                        db_label=task_config.db_config.db_label,
                        version=task_config.db_config.version,
                        note=task_config.db_config.note,
                        params=task_config.db_case_config.dict(),
                        case_name=case.name,
                        dataset=dataset.full_name,
                        dim=dataset.dim,
                        filter_type=filter_.type.name,
                        filter_rate=filter_.filter_rate,
                        k=task_config.case_config.k,
                        **metrics,
                    ).dict()
                )
    return results
