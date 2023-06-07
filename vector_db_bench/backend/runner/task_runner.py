import logging

from .case_runner import RunningStatus, CaseRunner
from .. import utils
from ...base import BaseModel

log = logging.getLogger(__name__)


class TaskRunner(BaseModel):
    run_id: str
    task_label: str
    case_runners: list[CaseRunner]

    def num_cases(self) -> int:
        return len(self.case_runners)

    def num_finished(self) -> int:
        return self._get_num_by_status(RunningStatus.FINISHED)

    def set_finished(self, idx: int) -> None:
        self.case_runners[idx].status = RunningStatus.FINISHED

    def _get_num_by_status(self, status: RunningStatus) -> int:
        return sum([1 for c in self.case_runners if c.status == status])

    def display(self) -> None:
        DATA_FORMAT = (" %34s | %-12s %-20s %7s | %-10s")
        TITLE_FORMAT = (" %s | %-12s %-20s %7s | %-10s") % (
            "DB", "CaseType", "Dataset", "Filter", "task_label")

        fmt = [TITLE_FORMAT]
        fmt.append(DATA_FORMAT%(
            "-"*11,
            "-"*12,
            "-"*20,
            "-"*7,
            "-"*7
        ))

        for f in self.case_runners:
            if f.ca.filter_rate != 0.0:
                filters = f.ca.filter_rate
            elif f.ca.filter_size != 0:
                filters = f.ca.filter_size
            else:
                filters = "None"

            ds_str = f"{f.ca.dataset.data.name}-{f.ca.dataset.data.label}-{utils.numerize(f.ca.dataset.data.size)}"
            fmt.append(DATA_FORMAT%(
                f.config.db_name,
                f.ca.label.name,
                ds_str,
                filters,
                self.task_label,
            ))

        log.info('\n'.join(fmt))
