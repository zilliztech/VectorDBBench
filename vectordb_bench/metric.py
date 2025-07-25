import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Metric:
    """result metrics"""

    # for load cases
    max_load_count: int = 0

    # for both performace and streaming cases
    insert_duration: float = 0.0
    optimize_duration: float = 0.0
    load_duration: float = 0.0  # insert + optimize

    # for performance cases
    qps: float = 0.0
    serial_latency_p99: float = 0.0
    serial_latency_p95: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0
    conc_num_list: list[int] = field(default_factory=list)
    conc_qps_list: list[float] = field(default_factory=list)
    conc_latency_p99_list: list[float] = field(default_factory=list)
    conc_latency_p95_list: list[float] = field(default_factory=list)
    conc_latency_avg_list: list[float] = field(default_factory=list)

    # for streaming cases
    st_ideal_insert_duration: int = 0
    st_search_stage_list: list[int] = field(default_factory=list)
    st_search_time_list: list[float] = field(default_factory=list)
    st_max_qps_list_list: list[float] = field(default_factory=list)
    st_recall_list: list[float] = field(default_factory=list)
    st_ndcg_list: list[float] = field(default_factory=list)
    st_serial_latency_p99_list: list[float] = field(default_factory=list)
    st_serial_latency_p95_list: list[float] = field(default_factory=list)
    st_conc_failed_rate_list: list[float] = field(default_factory=list)


QURIES_PER_DOLLAR_METRIC = "QP$ (Quries per Dollar)"
LOAD_DURATION_METRIC = "load_duration"
SERIAL_LATENCY_P99_METRIC = "serial_latency_p99"
SERIAL_LATENCY_P95_METRIC = "serial_latency_p95"
MAX_LOAD_COUNT_METRIC = "max_load_count"
QPS_METRIC = "qps"
RECALL_METRIC = "recall"

metric_unit_map = {
    LOAD_DURATION_METRIC: "s",
    SERIAL_LATENCY_P99_METRIC: "ms",
    SERIAL_LATENCY_P95_METRIC: "ms",
    MAX_LOAD_COUNT_METRIC: "K",
    QURIES_PER_DOLLAR_METRIC: "K",
}

lower_is_better_metrics = [
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    SERIAL_LATENCY_P95_METRIC,
]

metric_order = [
    QPS_METRIC,
    RECALL_METRIC,
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    SERIAL_LATENCY_P95_METRIC,
    MAX_LOAD_COUNT_METRIC,
]


def isLowerIsBetterMetric(metric: str) -> bool:
    return metric in lower_is_better_metrics


def calc_recall(count: int, ground_truth: list[int], got: list[int]) -> float:
    recalls = np.zeros(count)
    for i, result in enumerate(got):
        if result in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)


def get_ideal_dcg(k: int):
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i + 2)

    return ideal_dcg


def calc_ndcg(ground_truth: list[int], got: list[int], ideal_dcg: float) -> float:
    dcg = 0
    ground_truth = list(ground_truth)
    for got_id in set(got):
        if got_id in ground_truth:
            idx = ground_truth.index(got_id)
            dcg += 1 / np.log2(idx + 2)
    return dcg / ideal_dcg
