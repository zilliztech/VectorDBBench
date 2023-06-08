import logging
import pathlib
from datetime import date
from typing import Self
from enum import Enum

import ujson

from .backend.clients import (
    DB,
    DBConfig,
    DBCaseConfig,
    IndexType,
)
from .base import BaseModel
from . import config
from .metric import Metric


log = logging.getLogger(__name__)


class CaseType(Enum):
    """
    Value will be displayed in UI
    """

    LoadLDim = "Capacity Test(Large-dim)"
    LoadSDim = "Capacity Test(Small-dim)"

    PerformanceLZero = "Search Performance Test(Large Dataset)"
    PerformanceMZero = "Search Performance Test(Medium Dataset)"
    PerformanceSZero = "Search Performance Test(Small Dataset)"

    PerformanceLLow = (
        "Filtering Search Performance Test (Large Dataset, Low Filtering Rate)"
    )
    PerformanceMLow = (
        "Filtering Search Performance Test (Medium Dataset, Low Filtering Rate)"
    )
    PerformanceSLow = (
        "Filtering Search Performance Test (Small Dataset, Low Filtering Rate)"
    )
    PerformanceLHigh = (
        "Filtering Search Performance Test (Large Dataset, High Filtering Rate)"
    )
    PerformanceMHigh = (
        "Filtering Search Performance Test (Medium Dataset, High Filtering Rate)"
    )
    PerformanceSHigh = (
        "Filtering Search Performance Test (Small Dataset, High Filtering Rate)"
    )


class CaseConfigParamType(Enum):
    """
    Value will be the key of CaseConfig.params and displayed in UI
    """

    IndexType = "IndexType"
    M = "M"
    EFConstruction = "efConstruction"
    EF = "ef"
    SearchList = "search_list"
    Nlist = "nlist"
    Nprobe = "nprobe"
    MaxConnections = "maxConnections"
    numCandidates = "num_candidates"


class CustomizedCase(BaseModel):
    pass


class CaseConfig(BaseModel):
    """cases, dataset, test cases, filter rate, params"""
    case_id: CaseType
    custom_case: dict | None = None


class TaskConfig(BaseModel):
    db: DB
    db_config: DBConfig
    db_case_config: DBCaseConfig
    case_config: CaseConfig

    @property
    def db_name(self):
        db = self.db.value
        db_label = self.db_config.db_label
        return f"{db}-{db_label}" if db_label else db

class ResultLabel(Enum):
    NORMAL = ":)"
    FAILED = "x"
    OUTOFRANGE = "?"


class CaseResult(BaseModel):
    metrics: Metric
    task_config: TaskConfig
    label: ResultLabel = ResultLabel.NORMAL


class TestResult(BaseModel):
    """ ROOT/result_{date.today()}_{task_label}.json """
    run_id: str
    task_label: str
    results: list[CaseResult]

    def write_file(self):
        result_dir = config.RESULTS_LOCAL_DIR
        if not result_dir.exists():
            log.info(f"local result directory not exist, creating it: {result_dir}")
            result_dir.mkdir(parents=True)

        file_name = f'result_{date.today().strftime("%Y%m%d")}_{self.task_label}.json'
        result_file = result_dir.joinpath(file_name)
        if result_file.exists():
            log.warning(f"Replacing existing result with the same file_name: {result_file}")

        log.info(f"write results to disk {result_file}")
        with open(result_file, 'w') as f:
            b = self.json(exclude={'db_config': {'password', 'api_key'}})
            f.write(b)

    @classmethod
    def read_file(cls, full_path: pathlib.Path) -> Self:
        if not full_path.exists():
            raise ValueError(f"No such file: {full_path}")

        with open(full_path) as f:
            test_result = ujson.loads(f.read())
            if "task_label" not in test_result:
                test_result['task_label'] = test_result['run_id']

            for case_result in test_result["results"]:
                task_config = case_result.get("task_config")
                db = DB(task_config.get("db"))
                dbcls = db.init_cls
                task_config["db_config"] = dbcls.config_cls()(**task_config["db_config"])
                task_config["db_case_config"] = dbcls.case_config_cls(
                    index_type=task_config["db_case_config"].get("index", None),
                )(**task_config["db_case_config"])

                case_result["task_config"] = task_config
            c = TestResult.validate(test_result)
            return c

    def display(self, dbs: list[DB] | None = None):
        DATA_FORMAT = (" %-14s | %-17s %-20s %14s | %-10s %14s %14s %14s %14s")
        TITLE_FORMAT = (" %-14s | %-17s %-20s %14s | %-10s %14s %14s %14s %14s") % (
            "DB", "db_label", "case", "label", "load_dur", "qps", "latency(p99)", "recall", "max_load_count")

        SUMMERY_FORMAT = ("Task summery: run_id=%s, task_label=%s") % (
            self.run_id[:5], self.task_label)

        fmt = [SUMMERY_FORMAT, TITLE_FORMAT]
        fmt.append(DATA_FORMAT%(
            "-"*14,
            "-"*17,
            "-"*20,
            "-"*14,
            "-"*10,
            "-"*14,
            "-"*14,
            "-"*14,
            "-"*14,
        ))

        filter_list = dbs if dbs and isinstance(dbs, list) else None

        sorted_results = sorted(self.results, key=lambda x: (
            x.task_config.db.name,
            x.task_config.db_config.db_label,
            x.task_config.case_config.case_id.name,
        ), reverse=True)
        for f in sorted_results:
            if filter_list and f.task_config.db not in filter_list:
                continue

            fmt.append(DATA_FORMAT%(
                f.task_config.db.name,
                f.task_config.db_config.db_label,
                f.task_config.case_config.case_id.name,
                self.task_label,
                f.metrics.load_duration,
                f.metrics.qps,
                f.metrics.serial_latency_p99,
                f.metrics.recall,
                f.metrics.max_load_count,
            ))

        tmp_logger = logging.getLogger("no_color")
        for f in fmt:
            tmp_logger.info(f)
