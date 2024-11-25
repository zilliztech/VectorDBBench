import logging
import numpy as np

from dataclasses import dataclass, field


log = logging.getLogger(__name__)


@dataclass
class Metric:
    """result metrics"""

    # for load cases
    max_load_count: int = 0

    # for performance cases
    load_duration: float = 0.0  # duration to load all dataset into DB
    qps: float = 0.0
    serial_latency_p99: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0
    conc_num_list: list[int] = field(default_factory=list)
    conc_qps_list: list[float] = field(default_factory=list)
    conc_latency_p99_list: list[float] = field(default_factory=list)
    conc_latency_avg_list: list[float] = field(default_factory=list)


QURIES_PER_DOLLAR_METRIC = "QP$ (Quries per Dollar)"
LOAD_DURATION_METRIC = "load_duration"
SERIAL_LATENCY_P99_METRIC = "serial_latency_p99"
MAX_LOAD_COUNT_METRIC = "max_load_count"
QPS_METRIC = "qps"
RECALL_METRIC = "recall"

metricUnitMap = {
    LOAD_DURATION_METRIC: "s",
    SERIAL_LATENCY_P99_METRIC: "ms",
    MAX_LOAD_COUNT_METRIC: "K",
    QURIES_PER_DOLLAR_METRIC: "K",
}

lowerIsBetterMetricList = [
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
]

metricOrder = [
    QPS_METRIC,
    RECALL_METRIC,
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    MAX_LOAD_COUNT_METRIC,
]


def isLowerIsBetterMetric(metric: str) -> bool:
    return metric in lowerIsBetterMetricList


def calc_recall(count: int, ground_truth: list[int], got: list[int]) -> float:
    recalls = np.zeros(count)
    for i, result in enumerate(got):
        if result in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)


def get_ideal_dcg(k: int):
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i+2)

    return ideal_dcg


def calc_ndcg(ground_truth: list[int], got: list[int], ideal_dcg: float) -> float:
    dcg = 0
    ground_truth = list(ground_truth)
    for id in set(got):
        if id in ground_truth:
            idx = ground_truth.index(id)
            dcg += 1 / np.log2(idx+2)
    return dcg / ideal_dcg
