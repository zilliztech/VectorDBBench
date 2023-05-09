from ..results import TestResult, CaseResult


class ResultCollector:
    def get_results(self, path_to_results_folder: str) -> list[TestResult]:
        pass

    def _gen_test_result(path_to_file: str) -> TestResult:
        pass

    def _gen_case_result(config) -> CaseResult:
        pass
