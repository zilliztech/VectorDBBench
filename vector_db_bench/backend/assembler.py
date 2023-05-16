"""Assembler assembles cases with datasets and runners"""

from .cases import type2case, Case
from ..models import TaskConfig

class Assembler:

    @classmethod
    def assemble(cls, run_id: int, task: TaskConfig) -> Case:
        c_cls = type2case.get(task.case_config.case_id)

        c = c_cls(run_id=run_id)
        task.db_case_config.metric_type = c.dataset.data.metric_type

        db = task.db.init_cls(
            db_config=task.db_config.to_dict(),
            db_case_config=task.db_case_config,
            drop_old=True,
        )

        c.db = db
        return c
