import logging
import uuid
import concurrent

import psutil

from . import RESULTS_LOCAL_DIR
from .models import TaskConfig, TestResult, CaseResult
from .backend.result_collector import ResultCollector
from .backend.assembler import Assembler


log = logging.getLogger("__name__")

global_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
global_result_future: concurrent.futures.Future | None = None


class BenchMarkRunner:
    def __init__(self):
        self.running_task: dict | None = None
        self.latest_error: str | None = None

    def run(self, tasks: list[TaskConfig]) -> bool:
        """run all the tasks in the configs, write one result into the path"""
        if self.running_task is not None:
            log.warning("There're still tasks running in the background")
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
            'tasks': tasks,
            'progress': [False for i in range(len(tasks))],
        }

        return self._run_async()

    def get_results(self, result_dir: list[str] | None = None) -> list[TestResult]:
        """results of all runs, each TestResult represents one run."""
        target_dir = result_dir if result_dir else RESULTS_LOCAL_DIR
        return ResultCollector.collect(target_dir)

    def has_running(self) -> bool:
        """check if there're running benchmarks"""
        return self.running_task is not None

    def stop_running(self):
        """force stop if ther're running benchmarks"""
        self._clear_running_task()

    def get_tasks_count(self) -> int:
        """the count of all tasks"""
        if self.running_task:
            return len(self.running_task['cases'])
        return 0

    def get_current_task_id(self) -> int:
        """ the index of current running task
        return -1 if not running
        """
        if self.running_task:
            return self.running_task['run_id']
        return -1

    def _sync_running_task(self):
        if not self.running_task:
            return

        global global_result_future
        try:
            if global_result_future:
                global_result_future.result()
        except Exception as e:
            log.warning(f"task running failed: {e}", exc_info=True)
        finally:
            global_result_future = None
            self.running_task = 0


    def _async_task(self) -> None:
        if not self.running_task:
            return

        c_results = []
        for idx, c in enumerate(self.running_task['cases']):

            try:
                log.info(f"start running case: {c.model_dump(exclude=['dataset'])}")
                metric = c.run()
                log.info(f"end running case: {c.model_dump(exclude=['dataset'])}")

                c_results.append(CaseResult(
                    run_id=self.running_task['run_id'],
                    result_id=idx,
                    metrics=metric,
                    task_config=self.running_task['tasks'][idx],
                ))
                self.running_task['progress'][idx] = True

                test_result = TestResult(
                    run_id=self.running_task['run_id'],
                    results=c_results,
                )

                log.info(f"Write results file for task: {test_result}")
                test_result.write_file()
                log.info(f"Succes to finish task: {self.running_task['run_id']}")
            except Exception as e:
                err_msg = f"An error occurs when running case={c.model_dump(exclude=['dataset'])}, err={str(e)}"
                log.warning(err_msg)
                self.latest_error = err_msg
            finally:
                self.running_task = None


    def _clear_running_task(self):
        global global_result_future
        global_result_future = None

        if self.running_task:
            log.info(f"force stop running task: {self.running_task['run_id']}")
            for c in self.running_task['cases']:
                c.stop()

            for child_p in psutil.Process().children(recursive=True):
                log.warning(f"force killing child process: {child_p}")
                child_p.kill()

            self.running_task = None


    def _run_async(self) -> bool:
        log.info(f"task submitted: {self.running_task}")
        global global_result_future
        global_result_future = global_executor.submit(self._async_task, self.running_task)

        return True


benchMarkRunner = BenchMarkRunner()
