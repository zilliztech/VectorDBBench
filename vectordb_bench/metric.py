import logging
import numpy as np

from dataclasses import dataclass


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
