"""Assembler assembles cases with datasets and runners"""

from .cases import type2case, Case, CaseLabel
from ..models import TaskConfig

class Assembler:

    @classmethod
    def assemble(cls, run_id , task: TaskConfig) -> Case:
        c_cls = type2case.get(task.case_config.case_id)

        c = c_cls()
        task.db_case_config.metric_type = c.dataset.data.metric_type

        c.db_configs = (
            task.db.init_cls,
            task.db_config.to_dict(),
            task.db_case_config
        )
        return c

    @classmethod
    def assemble_all(cls, run_id: str, tasks: list[TaskConfig]) -> list[Case]:
        """group by case type, db, and case dataset"""
        cases = [cls.assemble(run_id, task) for task in tasks]
        load_cases = [c for c in cases if c.label == CaseLabel.Load]
        perf_cases = [c for c in cases if c.label == CaseLabel.Performance]

        # group by db
        db2cases = {}
        for c in perf_cases:
            db = c.db_configs[0]
            if db not in db2cases:
                db2cases[db] = []
            db2cases[db].append(c)

        for k in db2cases.keys():
            db2cases[k].sort(key=lambda x:x.dataset.data.size)
        return (load_cases, db2cases)
