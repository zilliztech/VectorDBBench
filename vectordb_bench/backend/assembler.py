from .cases import CaseLabel
from .task_runner import CaseRunner, RunningStatus, TaskRunner
from ..models import TaskConfig
from ..backend.clients import EmptyDBCaseConfig
import logging


log = logging.getLogger(__name__)


class Assembler:
    @classmethod
    def assemble(cls, run_id , task: TaskConfig) -> CaseRunner:
        c_cls = task.case_config.case_id.case_cls

        c = c_cls()
        if type(task.db_case_config) != EmptyDBCaseConfig:
            task.db_case_config.metric_type = c.dataset.data.metric_type

        runner = CaseRunner(
            run_id=run_id,
            config=task,
            ca=c,
            status=RunningStatus.PENDING,
        )

        return runner

    @classmethod
    def assemble_all(cls, run_id: str, task_label: str, tasks: list[TaskConfig]) -> TaskRunner:
        """group by case type, db, and case dataset"""
        runners = [cls.assemble(run_id, task) for task in tasks]
        load_runners = [r for r in runners if r.ca.label == CaseLabel.Load]
        perf_runners = [r for r in runners if r.ca.label == CaseLabel.Performance]

        # group by db
        db2runner = {}
        for r in perf_runners:
            db = r.config.db
            if db not in db2runner:
                db2runner[db] = []
            db2runner[db].append(r)

        # sort by dataset size
        for k in db2runner.keys():
            db2runner[k].sort(key=lambda x:x.ca.dataset.data.size)

        all_runners = []
        all_runners.extend(load_runners)
        for v in db2runner.values():
            all_runners.extend(v)

        return TaskRunner(
            run_id=run_id,
            task_label=task_label,
            case_runners=all_runners,
        )
