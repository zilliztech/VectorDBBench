import traceback
import logging
import uuid
import concurrent
import multiprocessing as mp

import psutil
from enum import Enum

from . import RESULTS_LOCAL_DIR
from .models import TaskConfig, TestResult, CaseResult
from .backend.result_collector import ResultCollector
from .backend.assembler import Assembler


log = logging.getLogger("__name__")

global_result_future: concurrent.futures.Future | None = None

class SIGNAL(Enum):
    SUCCESS=0
    ERROR=1
    WIP=2


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
        run_id = uuid.uuid4().hex
        log.info(f"generated uuid for the tasks: {run_id}")

        self.receive_conn, send_conn = mp.Pipe()
        self.latest_error = ""
        self.running_task = {
            'run_id': run_id,
            'cases': [Assembler.assemble(run_id, task) for task in tasks],
            'tasks': tasks,
            'progress': [False for i in range(len(tasks))],
        }

        return self._run_async(send_conn)

    def get_results(self, result_dir: list[str] | None = None) -> list[TestResult]:
        """results of all runs, each TestResult represents one run."""
        target_dir = result_dir if result_dir else RESULTS_LOCAL_DIR
        return ResultCollector.collect(target_dir)

    def _try_get_signal(self):
        if self.receive_conn and self.receive_conn.poll():
            sig, received = self.receive_conn.recv()
            log.debug(f"Sigal received to process: {sig}, {received}")
            if sig == SIGNAL.ERROR:
                self.latest_error = received
                self._clear_running_task()
            elif sig == SIGNAL.SUCCESS:
                global global_result_future
                global_result_future = None
                self.running_task = None
                self.receive_conn = None
            elif sig == SIGNAL.WIP:
                self.running_task['progress'][received] = True
            else:
                self._clear_running_task()

    def has_running(self) -> bool:
        """check if there're running benchmarks"""
        if self.running_task:
            self._try_get_signal()
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
        if not self.running_task:
            return -1

        return sum(self.running_task['progress'])

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
            self.running_task = None


    def _async_task(self, running_task: dict, send_conn: mp.connection.Connection) -> None:
        if not running_task:
            return

        c_results = []
        for idx, c in enumerate(running_task['cases']):

            try:
                log.info(f"start running case: {c.model_dump(exclude=['dataset'])}")
                metric = c.run()
                log.info(f"end running case: {c.model_dump(exclude=['dataset'])}")

                c_results.append(CaseResult(
                    result_id=idx,
                    metrics=metric,
                    task_config=running_task['tasks'][idx],
                ))

                send_conn.send((SIGNAL.WIP, idx))

            except Exception as e:
                err_msg = f"An error occurs when running case={c.model_dump(exclude=['dataset'])}, err={e}"
                traceback.print_exc()
                log.warning(err_msg)
                send_conn.send((SIGNAL.ERROR, err_msg))
                send_conn.close()
                return

        test_result = TestResult(
            run_id=self.running_task['run_id'],
            results=c_results,
        )

        log.info(f"Write results file for task: {test_result}")
        test_result.write_file()
        send_conn.send((SIGNAL.SUCCESS, None))
        send_conn.close()
        log.info(f"Succes to finish task: {self.running_task['run_id']}")


    def _clear_running_task(self):
        global global_result_future
        global_result_future = None

        if self.running_task:
            log.info(f"will force stop running task: {self.running_task['run_id']}")
            for c in self.running_task['cases']:
                c.stop()

            for child_p in psutil.Process().children(recursive=True):
                log.warning(f"force killing child process: {child_p}")
                child_p.kill()
            self.running_task = None

        if self.receive_conn:
            self.receive_conn.close()
            self.receive_conn = None


    def _run_async(self, conn: mp.connection.Connection) -> bool:
        log.info(f"task submitted: {self.running_task}")
        global global_result_future
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        global_result_future = executor.submit(self._async_task, self.running_task, conn)

        return True


benchMarkRunner = BenchMarkRunner()
