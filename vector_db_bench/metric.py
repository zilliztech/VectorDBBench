import logging
import numpy as np

from dataclasses import dataclass


log  = logging.getLogger(__name__)

@dataclass
class Metric:
    """result metrics"""

    # for load cases
    max_load_count: int = 0

    # for performance cases
    load_duration: float = 0. # duration to load all dataset into DB
    build_duration: float = 0. # duration of ready_search
    qps: float = 0.
    serial_latency: float = 0.
    p99: float = 0.
    recall: float = 0.


def calc_recall(count: int, ground_truth: list[int], got: list[tuple[int, float]]) -> float:
    recalls = np.zeros(count)
    for i, result  in enumerate(got):
        if result[0] in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)
