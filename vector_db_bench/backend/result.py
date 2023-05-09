from pydantic import BaseModel
from typing import Any

from .metric import Metric

class CaseResult(BaseModel):
    result_id: int
    case_id: int
    case_config: Any
    output_path: str

    metrics: list[Metric]

    def append_to_disk(self):
        pass

class TestResult(BaseModel):
    run_id: int
    results: list[CaseResult]


