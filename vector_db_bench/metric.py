from pydantic import BaseModel


class Metric(BaseModel):
    qps: float
    recall: float
