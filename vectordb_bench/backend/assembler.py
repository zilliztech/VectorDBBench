import logging

from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.models import TaskConfig

from .cases import CaseLabel
from .dataset import FtsDatasetManager
from .task_runner import CaseRunner, RunningStatus, TaskRunner

log = logging.getLogger(__name__)


class FilterNotSupportedError(ValueError):
    """Raised when a filter type is not supported by a vector database."""

    def __init__(self, db_name: str, filter_type: FilterOp):
        super().__init__(f"{filter_type} Filter test is not supported by {db_name}.")


class Assembler:
    @classmethod
    def assemble(cls, run_id: str, task: TaskConfig, source: DatasetSource) -> CaseRunner:
        c_cls = task.case_config.case_id.case_cls

        c = c_cls(task.case_config.custom_case)
        actual_source = DatasetSource.IR_DATASETS if isinstance(c.dataset, FtsDatasetManager) else source

        if (
            type(task.db_case_config) is not EmptyDBCaseConfig
            and not isinstance(c.dataset, FtsDatasetManager)
            and hasattr(c.dataset.data, "metric_type")
        ):
            task.db_case_config.metric_type = c.dataset.data.metric_type

        return CaseRunner(
            run_id=run_id,
            config=task,
            ca=c,
            status=RunningStatus.PENDING,
            dataset_source=actual_source,
        )

    @classmethod
    def assemble_all(
        cls,
        run_id: str,
        task_label: str,
        tasks: list[TaskConfig],
        source: DatasetSource,
    ) -> TaskRunner:
        """group by case type, db, and case dataset"""
        runners = [cls.assemble(run_id, task, source) for task in tasks]
        load_runners = [r for r in runners if r.ca.label == CaseLabel.Load]
        perf_runners = [r for r in runners if r.ca.label == CaseLabel.Performance]
        streaming_runners = [r for r in runners if r.ca.label == CaseLabel.Streaming]
        cloud_insert_runners = [r for r in runners if r.ca.label == CaseLabel.CloudInsert]
        cloud_cold_latency_runners = [r for r in runners if r.ca.label == CaseLabel.CloudColdLatency]
        fts_runners = [r for r in runners if r.ca.label == CaseLabel.FullTextSearchPerformance]

        search_filter_runners = [*perf_runners, *cloud_cold_latency_runners]

        # group by db
        db2runner: dict[DB, list[CaseRunner]] = {}
        for r in search_filter_runners:
            db = r.config.db
            if db not in db2runner:
                db2runner[db] = []
            db2runner[db].append(r)

        # check
        for db, runners in db2runner.items():
            db_instance = db.init_cls
            for runner in runners:
                if not db_instance.filter_supported(runner.ca.filters):
                    raise FilterNotSupportedError(db.value, runner.ca.filters.type)

        # sort by dataset size
        for _, runner in db2runner.items():
            runner.sort(key=lambda x: (x.ca.dataset.data.size, 0 if x.ca.filters.type == FilterOp.StrEqual else 1))

        all_runners = []
        all_runners.extend(load_runners)
        all_runners.extend(streaming_runners)
        all_runners.extend(cloud_insert_runners)
        for v in db2runner.values():
            all_runners.extend(v)
        all_runners.extend(fts_runners)

        return TaskRunner(
            run_id=run_id,
            task_label=task_label,
            case_runners=all_runners,
        )
