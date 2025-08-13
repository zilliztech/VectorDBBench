import json
import logging


from vectordb_bench.backend.cases import CaseType, StreamingPerformanceCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.models import CaseResult
from vectordb_bench import config
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

from vectordb_bench.interface import BenchMarkRunner


def get_standard_2025_results() -> list[CaseResult]:
    all_results = BenchMarkRunner.get_results()
    standard_2025_case_results = []
    for result in all_results:
        if result.task_label == "standard_2025":
            standard_2025_case_results += result.results
    return standard_2025_case_results


def save_to_json(data: list[dict], file_name: str):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)


def main():
    standard_2025_case_results = get_standard_2025_results()
    data = []
    streaming_data = []
    for case_result in standard_2025_case_results:
        db = case_result.task_config.db
        label = case_result.task_config.db_config.db_label
        db_name = f"{db.value}{f'-{label}' if label else ''}"
        metrics = case_result.metrics
        qps = metrics.qps
        latency = metrics.serial_latency_p99
        recall = metrics.recall
        case = case_result.task_config.case_config.case
        filter_ratio = case.filters.filter_rate
        dataset = case.dataset.data.full_name
        if case.case_id != CaseType.StreamingPerformanceCase:
            data.append(
                {
                    "dataset": dataset,
                    "db": db.value,
                    "label": label,
                    "db_name": db_name,
                    "qps": round(qps, 4),
                    "latency": round(latency, 4),
                    "recall": round(recall, 4),
                    "filter_ratio": round(filter_ratio, 3),
                }
            )
        else:
            case: StreamingPerformanceCase = case
            # use 90p search stage results to represent streaming performance
            qps_90p = metrics.st_max_qps_list_list[metrics.st_search_stage_list.index(90)]
            latency_90p = metrics.st_serial_latency_p99_list[metrics.st_search_stage_list.index(90)]
            insert_rate = case.insert_rate
            streaming_data.append(
                {
                    "dataset": dataset,
                    "db": db.value,
                    "label": label,
                    "db_name": db_name,
                    "insert_rate": insert_rate,
                    "streaming_qps": round(qps_90p, 4),
                    "streaming_latency": round(latency_90p, 4),
                }
            )
    save_to_json(data, config.RESULTS_LOCAL_DIR / "leaderboard_v2.json")
    save_to_json(streaming_data, config.RESULTS_LOCAL_DIR / "leaderboard_v2_streaming.json")


if __name__ == "__main__":
    main()
