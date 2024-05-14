import logging
import pathlib
from datetime import date
from enum import Enum, StrEnum, auto
from typing import List, Self, Sequence, Set

import ujson

from .backend.clients import (
    DB,
    DBConfig,
    DBCaseConfig,
    IndexType,
)
from .backend.cases import CaseType
from .base import BaseModel
from . import config
from .metric import Metric


log = logging.getLogger(__name__)


class LoadTimeoutError(TimeoutError):
    pass


class PerformanceTimeoutError(TimeoutError):
    pass


class CaseConfigParamType(Enum):
    """
    Value will be the key of CaseConfig.params and displayed in UI
    """

    IndexType = "IndexType"
    M = "M"
    EFConstruction = "efConstruction"
    ef_construction = "ef_construction"
    EF = "ef"
    SearchList = "search_list"
    ef_search = "ef_search"
    Nlist = "nlist"
    Nprobe = "nprobe"
    MaxConnections = "maxConnections"
    numCandidates = "num_candidates"
    lists = "lists"
    probes = "probes"
    quantizationType = "quantizationType"
    quantizationRatio = "quantizationRatio"
    m = "m"
    nbits = "nbits"
    intermediate_graph_degree = "intermediate_graph_degree"
    graph_degree = "graph_degree"
    itopk_size = "itopk_size"
    team_size = "team_size"
    search_width = "search_width"
    min_iterations = "min_iterations"
    max_iterations = "max_iterations"
    build_algo = "build_algo"
    cache_dataset_on_device = "cache_dataset_on_device"
    refine_ratio = "refine_ratio"
    level = "level"
    maintenance_work_mem = "maintenance_work_mem"
    max_parallel_workers = "max_parallel_workers"

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
    run_id: str
    task_label: str
    results: list[CaseResult]

    file_fmt: str = "result_{}_{}_{}.json"  # result_20230718_statndard_milvus.json

    def flush(self):
        db2case = self.get_db_results()

        result_root = config.RESULTS_LOCAL_DIR
        for db, result in db2case.items():
            self.write_db_file(
                result_dir=result_root.joinpath(db.value),
                partial=TestResult(
                    run_id=self.run_id, task_label=self.task_label, results=result
                ),
                db=db.value.lower(),
            )

    def get_db_results(self) -> dict[DB, CaseResult]:
        db2case = {}
        for res in self.results:
            if res.task_config.db in db2case:
                db2case[res.task_config.db].append(res)
            else:
                db2case[res.task_config.db] = [res]
        return db2case

    def write_db_file(self, result_dir: pathlib.Path, partial: Self, db: str):
        if not result_dir.exists():
            log.info(f"local result directory not exist, creating it: {result_dir}")
            result_dir.mkdir(parents=True)

        file_name = self.file_fmt.format(
            date.today().strftime("%Y%m%d"), partial.task_label, db
        )
        result_file = result_dir.joinpath(file_name)
        if result_file.exists():
            log.warning(
                f"Replacing existing result with the same file_name: {result_file}"
            )

        log.info(f"write results to disk {result_file}")
        with open(result_file, "w") as f:
            b = partial.json(exclude={"db_config": {"password", "api_key"}})
            f.write(b)

    @classmethod
    def read_file(cls, full_path: pathlib.Path, trans_unit: bool = False) -> Self:
        if not full_path.exists():
            raise ValueError(f"No such file: {full_path}")

        with open(full_path) as f:
            test_result = ujson.loads(f.read())
            if "task_label" not in test_result:
                test_result["task_label"] = test_result["run_id"]

            for case_result in test_result["results"]:
                task_config = case_result.get("task_config")
                db = DB(task_config.get("db"))

                task_config["db_config"] = db.config_cls(**task_config["db_config"])
                task_config["db_case_config"] = db.case_config_cls(
                    index_type=task_config["db_case_config"].get("index", None),
                )(**task_config["db_case_config"])

                case_result["task_config"] = task_config

                if trans_unit:
                    cur_max_count = case_result["metrics"]["max_load_count"]
                    case_result["metrics"]["max_load_count"] = (
                        cur_max_count / 1000
                        if int(cur_max_count) > 0
                        else cur_max_count
                    )

                    cur_latency = case_result["metrics"]["serial_latency_p99"]
                    case_result["metrics"]["serial_latency_p99"] = (
                        cur_latency * 1000 if cur_latency > 0 else cur_latency
                    )
            c = TestResult.validate(test_result)

            return c

    def display(self, dbs: list[DB] | None = None):
        filter_list = dbs if dbs and isinstance(dbs, list) else None
        sorted_results = sorted(
            self.results,
            key=lambda x: (
                x.task_config.db.name,
                x.task_config.db_config.db_label,
                x.task_config.case_config.case_id.name,
            ),
            reverse=True,
        )

        filtered_results = [
            r
            for r in sorted_results
            if not filter_list or r.task_config.db not in filter_list
        ]

        def append_return(x, y):
            x.append(y)
            return x

        max_db = max(map(len, [f.task_config.db.name for f in filtered_results]))
        max_db_labels = (
            max(map(len, [f.task_config.db_config.db_label for f in filtered_results]))
            + 3
        )
        max_case = max(
            map(len, [f.task_config.case_config.case_id.name for f in filtered_results])
        )
        max_load_dur = (
            max(map(len, [str(f.metrics.load_duration) for f in filtered_results])) + 3
        )
        max_qps = max(map(len, [str(f.metrics.qps) for f in filtered_results])) + 3
        max_recall = (
            max(map(len, [str(f.metrics.recall) for f in filtered_results])) + 3
        )

        max_db_labels = 8 if max_db_labels < 8 else max_db_labels
        max_load_dur = 11 if max_load_dur < 11 else max_load_dur
        max_qps = 10 if max_qps < 10 else max_qps
        max_recall = 13 if max_recall < 13 else max_recall

        LENGTH = (
            max_db,
            max_db_labels,
            max_case,
            len(self.task_label),
            max_load_dur,
            max_qps,
            15,
            max_recall,
            14,
            5,
        )

        DATA_FORMAT = (
            f"%-{max_db}s | %-{max_db_labels}s %-{max_case}s %-{len(self.task_label)}s"
            f" | %-{max_load_dur}s %-{max_qps}s %-15s %-{max_recall}s %-14s"
            f" | %-5s"
        )

        TITLE = DATA_FORMAT % (
            "DB",
            "db_label",
            "case",
            "label",
            "load_dur",
            "qps",
            "latency(p99)",
            "recall",
            "max_load_count",
            "label",
        )
        SPLIT = DATA_FORMAT % tuple(map(lambda x: "-" * x, LENGTH))
        SUMMARY_FORMAT = ("Task summary: run_id=%s, task_label=%s") % (
            self.run_id[:5],
            self.task_label,
        )
        fmt = [SUMMARY_FORMAT, TITLE, SPLIT]

        for f in filtered_results:
            fmt.append(
                DATA_FORMAT
                % (
                    f.task_config.db.name,
                    f.task_config.db_config.db_label,
                    f.task_config.case_config.case_id.name,
                    self.task_label,
                    f.metrics.load_duration,
                    f.metrics.qps,
                    f.metrics.serial_latency_p99,
                    f.metrics.recall,
                    f.metrics.max_load_count,
                    f.label.value,
                )
            )

        tmp_logger = logging.getLogger("no_color")
        for f in fmt:
            tmp_logger.info(f)
