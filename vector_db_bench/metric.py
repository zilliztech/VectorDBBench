import logging
import numpy as np
from pydantic import BaseModel


log  = logging.getLogger(__name__)
class Metric(BaseModel):
    """result metrics"""

    load_time: float = 0.0
    max_load_count: int = 0
    qps: float = 0
    recall: float = 0
    serial_latency: float = 0
    load_duration: float = 0
    build_duration: float = 0


def calc_recall(count: int, ground_truth: list[int], got: list[tuple[int, float]]):
    recalls = np.zeros(count)
    for i, result  in enumerate(got):
        if result[0] in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)
