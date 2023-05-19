"""Assembler assembles cases with datasets and runners"""

from .cases import type2case, Case
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
