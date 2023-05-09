from pydantic import BaseModel
from .models import TaskConfig, TestResult
from .backend.result_collector import ResultCollector
from .backend.assembler import Assembler


class BenchMarkRunner(BaseModel):
    result_collector: ResultCollector
    assembler: Assembler

    def run(configs: list[TaskConfig]) -> bool:
        """run all the tasks in the configs, write one result into the path"""
        pass

    def get_results(paths: list[str]) -> list[TestResult]:
        """results of all runs, each TestResult represnets one run."""
        pass

    def has_running() -> bool:
        """check if there're running benchmarks"""
        pass

    def stop_running():
        """force stop if ther're running benchmarks"""
        pass

