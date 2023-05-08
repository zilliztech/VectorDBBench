from .metrics import Metric
from typing import List


class TestResult:
    run_id: int
    results: List[CaseResult]


class CaseResult:
    result_id: int
    case_id: int
    case_config: Any
    output_path: str

    metrics: List[Metric]

    def append_to_disk(self):
        pass
