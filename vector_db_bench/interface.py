import logging
import uuid
import concurrent
from typing import Any

from pydantic import BaseModel, ConfigDict

from .models import TaskConfig, TestResult
from .backend.cases import Case
from .backend.result_collector import ResultCollector
from .backend.assembler import Assembler


log = logging.getLogger("__name__")

global_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
global_result_future: concurrent.futures.Future | None = None


class BenchMarkRunner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_collector: ResultCollector | None = None

    running_task: dict | None = None
    result_future: Any | None = None

    def run(self, tasks: list[TaskConfig]) -> bool:
        """run all the tasks in the configs, write one result into the path"""
        if self.running_task is not None:
            log.warning("There're still tasks running in the background")
            #  TODO remove, test stop_running
            #  self.stop_running()
            #  if global_result_future:
            #      concurrent.futures.wait([global_result_future])
            #      r = global_result_future.result()
            #      log.info(f"yx: {r}")
            return False

        if len(tasks) == 0:
            log.warning("Empty tasks submitted")
            return False

        log.debug(f"tasks: {tasks}")

        # Generate run_id
        run_id = uuid.uuid4().int
        self.running_task = {
            'run_id': run_id,
            'cases': [Assembler.assemble(run_id, task) for task in tasks],
            'progress': [False for i in range(len(tasks))],
        }

        return self._run_async()


    def get_results(paths: list[str]) -> list[TestResult]:
        """results of all runs, each TestResult represnets one run."""
        pass

    def has_running(self) -> bool:
        """check if there're running benchmarks"""
        return self.running_task is not None

    def stop_running(self):
        """force stop if ther're running benchmarks"""
        if self.running_task:
            for c in self.running_task['cases']:
                c.stop()

            log.info(f"force stopped running task: {self.running_task['run_id']}")
            self.running_task = None

    def get_tasks_count(self) -> int:
        """the count of all tasks"""
        if self.running_task:
            return len(self.running_task['cases'])
        return 0

    def get_current_task_id(self) -> int:
        """
        the index of current running task
        return -1 if not running
        """
        if self.running_task:
            return self.running_task['run_id']
        return -1

    def _async_task(self):
        if not self.running_task:
            return

        for idx, c in enumerate(self.running_task['cases']):
            log.info(f"start running case: {c.model_dump(exclude=['dataset'])}")
            c.run()
            self.running_task['progress'][idx] = True
            log.info(f"end running case: {c.model_dump(exclude=['dataset'])}")

        self.running_task = None


    def _run_async(self) -> bool:
        log.info(f"task submitted: {self.running_task}")
        global global_result_future
        global_result_future = global_executor.submit(self._async_task)

        return True
