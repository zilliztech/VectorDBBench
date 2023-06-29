import pathlib
from ..models import TestResult


class ResultCollector:
    @classmethod
    def collect(cls, result_dir: pathlib.Path) -> list[TestResult]:
        results = []
        if not result_dir.exists() or len(list(result_dir.glob("result_*.json"))) == 0:
            return []

        for json_file in result_dir.glob("result_*.json"):
            results.append(TestResult.read_file(json_file, trans_unit=True))

        return results
