import logging

from vectordb_bench.backend.clients import EmptyDBCaseConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.models import TaskConfig

from .cases import CaseLabel
from .task_runner import CaseRunner, RunningStatus, TaskRunner

log = logging.getLogger(__name__)


class Assembler:
    @classmethod
    def assemble(cls, run_id: str, task: TaskConfig, source: DatasetSource) -> CaseRunner:
        c_cls = task.case_config.case_id.case_cls

        c = c_cls(task.case_config.custom_case)
        if type(task.db_case_config) is not EmptyDBCaseConfig:
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

        # group by db
        db2runner = {}
        for r in perf_runners:
            db = r.config.db
            if db not in db2runner:
                db2runner[db] = []
            db2runner[db].append(r)

        # check dbclient installed
        for k in db2runner:
            _ = k.init_cls

        # sort by dataset size
        for _, runner in db2runner.items():
            runner.sort(key=lambda x: x.ca.dataset.data.size)

        all_runners = []
        all_runners.extend(load_runners)
        for v in db2runner.values():
            all_runners.extend(v)

        return TaskRunner(
            run_id=run_id,
            task_label=task_label,
            case_runners=all_runners,
        )
