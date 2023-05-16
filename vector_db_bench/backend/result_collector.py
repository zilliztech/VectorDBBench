import pathlib
from ..models import TestResult


class ResultCollector:
    @classmethod
    def collect(cls, target_dir: str) -> list[TestResult]:
        results = []
        result_dir = pathlib.Path(target_dir)
        if not result_dir.exists() or len(list(result_dir.glob("*.json"))) == 0:
            return []

    
        for json_file in result_dir.glob("*.json"):
            results.append(TestResult.read_file(json_file))

        return results
