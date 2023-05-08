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
    QPS: float
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

-> BaseConfig and DBConfig can merge into DBConfig


### APIs
```python

def run(configs: List[TaskConfig], path: str) -> int:
    """run all the tasks in the configs, write one result into the path?

    Examples:
        dbs = [db_a, db_b]
        cases = [case_x, case_y, case_z]

        configs = dbs X cases
                = ((db, case) for db in dbs
                              for case in cases)
                = ((db_a, case_x), (db_a, case_y), (db_a, case_z),
                   (db_b, case_x'), (db_b, case_y'), (db_b, case_z'))

        TestResult = [CaseResult("db_a", "case_x"), ..., CaseResult("db_b", "case_z")]

    Returns:
        int: run_id
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
