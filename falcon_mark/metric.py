from pydantic import BaseModel


class Metric(BaseModel):
    """result metrics"""
    pass

class LoadMetric(BaseModel):
    label: str = "l_metric"
    load_count_max: int
    load_speed: float

class PerformanceMetric(BaseModel):
    label: str = "p_metric"

    qps: float
    recall: float
    latency: float # (pqq, p95)
    insert_duration: float
    build_duration: float
