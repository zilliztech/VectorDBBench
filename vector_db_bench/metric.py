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

metricUnitMap = {
    'load_duration': 's',
    'serial_latency_p99': 's',
}

lowerIsBetterMetricList = [
    "load_duration",
    "serial_latency_p99",
]

metricOrder = [
    "qps",
    "recall",
    "load_duration",
    "serial_latency_p99",
    "max_load_count",
]


def isLowerIsBetterMetric(metric: str) -> bool:
    return metric in lowerIsBetterMetricList


def calc_recall(count: int, ground_truth: list[int], got: list[int]) -> float:
    recalls = np.zeros(count)
    for i, result in enumerate(got):
        if result in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)
