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
        if (
            type(task.db_case_config) is not EmptyDBCaseConfig
            and not isinstance(c.dataset, FtsDatasetManager)
            and hasattr(c.dataset.data, "metric_type")
        ):
            # FTS datasets don't have metric_type, skip setting it
            task.db_case_config.metric_type = c.dataset.data.metric_type

        return CaseRunner(
            run_id=run_id,
            config=task,
            ca=c,
            status=RunningStatus.PENDING,
            dataset_source=source,
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
        fts_runners = [r for r in runners if r.ca.label == CaseLabel.FullTextSearchPerformance]

        # group by db
        db2runner: dict[DB, list[CaseRunner]] = {}
        for r in perf_runners:
            db = r.config.db
            if db not in db2runner:
                db2runner[db] = []
            db2runner[db].append(r)

        # group FTS runners by db
        db2fts_runner: dict[DB, list[CaseRunner]] = {}
        for r in fts_runners:
            db = r.config.db
            if db not in db2fts_runner:
                db2fts_runner[db] = []
            db2fts_runner[db].append(r)

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
        for v in db2runner.values():
            all_runners.extend(v)
        # Add FTS runners
        for v in db2fts_runner.values():
            all_runners.extend(v)

        return TaskRunner(
            run_id=run_id,
            task_label=task_label,
            case_runners=all_runners,
        )
