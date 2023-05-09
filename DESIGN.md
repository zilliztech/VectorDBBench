# Design details

Frond End -> API -> Back End

## APIs between front and end

### models
```python
# Input Configs
class DB(IntEnum):
    """types of db"""

class CaseConfig:
    """dataset, test cases, filter rate, params"""
    # TODO

class DBConfig:
    """DB authentications: host, port, user, password, and, token"""
    # TODO

TaskConfig = (DB, DBConfig, CaseConfig)

# Output Results
class Metric:
    # TODO

class CaseResult:
    result_id: int
    case_id: int
    case_config: CaseConfig
    output_path: str
    metrics: List[Metric]

class TestResult:
    run_id: int
    results: List[CaseResult]
```


### APIs
```python

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
```

## Back End Framework
