import logging
import pathlib

from vectordb_bench.models import TestResult

log = logging.getLogger(__name__)


class ResultCollector:
    @classmethod
    def collect(cls, result_dir: pathlib.Path) -> list[TestResult]:
        reg = "result_*.json"
        results_d = {}
        if not result_dir.exists() or len(list(result_dir.rglob(reg))) == 0:
            return []

        for json_file in result_dir.rglob(reg):
            file_result = TestResult.read_file(json_file, trans_unit=True)

            # Group result files of the same run_id into one TestResult
            if file_result.run_id in results_d:
                results_d[file_result.run_id].results.extend(file_result.results)
            else:
                results_d[file_result.run_id] = file_result

        return list(results_d.values())
