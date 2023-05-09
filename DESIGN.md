# Design details

Frond End -> API -> Back End

## APIs between front and end

### models
```python
# Input Configs
class BaseConfig:
    """types of db"""
    # TODO

class CaseConfig:
    """dataset, test cases, filter rate, params"""
    # TODO

class DBConfig:
    """DB authentications: host, port, user, password, and, token"""
    # TODO

TaskConfig = (BaseConfig, CaseConfig, DBConfig)

# Output Results
class Metric:
    # TODO
    qps: float
    recall: float
    latency: Any
    insert_duration: float
    build_duration: float

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

def run(configs: List[TaskConfig], path: str) -> bool:
    """run all the tasks in the configs, write one result into the path?

    """
    pass

def get_results(paths: List[str]) -> List[TestResult]:
    """results of all runs, each TestResult represents one run."""

def has_running() ->  bool:
    """check if there're running benchmarks"""

def stop_running():
    """force stop if there're running benchmarks"""
```

## Back End Framework
