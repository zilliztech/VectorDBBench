from typing import Any, List


def run(configs: List[Any]) -> bool:
    """run all the tasks in the configs, write one result into the path"""
    pass

def get_results(paths: List[str]) -> List[Any]:
    """results of all runs, each TestResult represnets one run."""
    pass

def has_running() -> bool:
    """check if there're running benchmarks"""
    pass

def stop_running():
    """force stop if ther're running benchmarks"""
    pass

