import concurrent.futures
import logging
import multiprocessing as mp
import pathlib
import signal
import traceback
import uuid
from collections.abc import Callable
from enum import Enum
from multiprocessing.connection import Connection

import psutil

from . import config
from .backend.assembler import Assembler
from .backend.data_source import DatasetSource
from .backend.result_collector import ResultCollector
from .backend.task_runner import TaskRunner
from .metric import Metric
from .models import (
    CaseResult,
    LoadTimeoutError,
    PerformanceTimeoutError,
    ResultLabel,
    TaskConfig,
    TaskStage,
    TestResult,
)

log = logging.getLogger(__name__)

global_result_future: concurrent.futures.Future | None = None


class SIGNAL(Enum):
    SUCCESS = 0
    ERROR = 1
    WIP = 2


class BenchMarkRunner:
    def __init__(self):
        self.running_task: TaskRunner | None = None
        self.latest_error: str | None = None
        self.drop_old: bool = True
        self.dataset_source: DatasetSource = DatasetSource.S3

    def set_drop_old(self, drop_old: bool):
        self.drop_old = drop_old

    def set_download_address(self, use_aliyun: bool):
        if use_aliyun:
            self.dataset_source = DatasetSource.AliyunOSS
        else:
            self.dataset_source = DatasetSource.S3

    def run(self, tasks: list[TaskConfig], task_label: str | None = None) -> bool:
        """run all the tasks in the configs, write one result into the path"""
        self.latest_error = ""
        if self.running_task is not None:
            log.warning("There're still tasks running in the background")
            return False

        if len(tasks) == 0:
            log.warning("Empty tasks submitted")
            return False

        log.debug(f"tasks: {tasks}, task_label: {task_label}, dataset source: {self.dataset_source}")

        # Generate run_id
        run_id = uuid.uuid4().hex
        log.info(f"generated uuid for the tasks: {run_id}")
        task_label = task_label if task_label else run_id

        self.receive_conn, send_conn = mp.Pipe()
        self.latest_error = ""

        try:
            self.running_task = Assembler.assemble_all(
                run_id,
                task_label,
                tasks,
                self.dataset_source,
            )
            self.running_task.display()
        except ModuleNotFoundError as e:
            msg = f"Please install client for database, error={e}"
            log.warning(msg)
            self.latest_error = msg
            return True

        return self._run_async(send_conn)

    def get_results(self, result_dir: pathlib.Path | None = None) -> list[TestResult]:
        """results of all runs, each TestResult represents one run."""
        target_dir = result_dir if result_dir else config.RESULTS_LOCAL_DIR
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
                self.running_task.set_finished(received)
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
            return self.running_task.num_cases()
        return 0

    def get_current_task_id(self) -> int:
        """the index of current running task
        return -1 if not running
        """
        if not self.running_task:
            return -1
        return self.running_task.num_finished()

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

    def _async_task_v2(self, running_task: TaskRunner, send_conn: Connection) -> None:
        try:
            if not running_task:
                return

            c_results = []
            latest_runner, cached_load_duration = None, None
            for idx, runner in enumerate(running_task.case_runners):
                case_res = CaseResult(
                    metrics=Metric(),
                    task_config=runner.config,
                )

                drop_old = TaskStage.DROP_OLD in runner.config.stages
                if (latest_runner and runner == latest_runner) or not self.drop_old:
                    drop_old = False
                num_cases = running_task.num_cases()
                try:
                    log.info(f"[{idx+1}/{num_cases}] start case: {runner.display()}, drop_old={drop_old}")
                    case_res.metrics = runner.run(drop_old)
                    log.info(
                        f"[{idx+1}/{num_cases}] finish case: {runner.display()}, "
                        f"result={case_res.metrics}, label={case_res.label}"
                    )

                    # cache the latest succeeded runner
                    latest_runner = runner

                    # cache the latest drop_old=True load_duration of the latest succeeded runner
                    cached_load_duration = case_res.metrics.load_duration if drop_old else cached_load_duration

                    # use the cached load duration if this case didn't drop the existing collection
                    if not drop_old:
                        case_res.metrics.load_duration = cached_load_duration if cached_load_duration else 0.0
                except (LoadTimeoutError, PerformanceTimeoutError) as e:
                    log.warning(f"[{idx+1}/{num_cases}] case {runner.display()} failed to run, reason={e}")
                    case_res.label = ResultLabel.OUTOFRANGE
                    continue

                except Exception as e:
                    log.warning(f"[{idx+1}/{num_cases}] case {runner.display()} failed to run, reason={e}")
                    traceback.print_exc()
                    case_res.label = ResultLabel.FAILED
                    continue

                finally:
                    c_results.append(case_res)
                    send_conn.send((SIGNAL.WIP, idx))

            test_result = TestResult(
                run_id=running_task.run_id,
                task_label=running_task.task_label,
                results=c_results,
            )
            test_result.display()
            test_result.flush()

            send_conn.send((SIGNAL.SUCCESS, None))
            send_conn.close()
            log.info(f"Success to finish task: label={running_task.task_label}, run_id={running_task.run_id}")

        except Exception as e:
            err_msg = (
                f"An error occurs when running task={running_task.task_label}, run_id={running_task.run_id}, err={e}"
            )
            traceback.print_exc()
            log.warning(err_msg)
            send_conn.send((SIGNAL.ERROR, err_msg))
            send_conn.close()
            return

    def _clear_running_task(self):
        global global_result_future
        global_result_future = None

        if self.running_task:
            log.info(f"will force stop running task: {self.running_task.run_id}")
            for r in self.running_task.case_runners:
                r.stop()

            self.kill_proc_tree(timeout=5)
            self.running_task = None

        if self.receive_conn:
            self.receive_conn.close()
            self.receive_conn = None

    def _run_async(self, conn: Connection) -> bool:
        log.info(
            f"task submitted: id={self.running_task.run_id}, {self.running_task.task_label}, "
            f"case number: {len(self.running_task.case_runners)}"
        )
        global global_result_future
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
        )
        global_result_future = executor.submit(self._async_task_v2, self.running_task, conn)

        return True

    def kill_proc_tree(
        self,
        sig: int = signal.SIGTERM,
        timeout: float | None = None,
        on_terminate: Callable | None = None,
    ):
        """Kill a process tree (including grandchildren) with signal
        "sig" and return a (gone, still_alive) tuple.
        "on_terminate", if specified, is a callback function which is
        called as soon as a child terminates.
        """
        children = psutil.Process().children(recursive=True)
        for p in children:
            try:
                log.warning(f"sending SIGTERM to child process: {p}")
                p.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(children, timeout=timeout, callback=on_terminate)

        for p in alive:
            log.warning(f"force killing child process: {p}")
            p.kill()


benchmark_runner = BenchMarkRunner()
