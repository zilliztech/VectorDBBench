import concurrent
import logging
import multiprocessing as mp
import random
import time
import traceback
from collections.abc import Iterable
from multiprocessing.queues import Queue

import numpy as np

from vectordb_bench.backend.filter import Filter, non_filter

from ... import config
from ...models import ConcurrencySlotTimeoutError
from ..clients import api

NUM_PER_BATCH = config.NUM_PER_BATCH
log = logging.getLogger(__name__)


class MultiProcessingSearchRunner:
    """multiprocessing search runner

    Args:
        k(int): search topk, default to 100
        concurrency(Iterable): concurrencies, default [1, 5, 10, 15, 20, 25, 30, 35]
        duration(int): duration for each concurency, default to 30s
    """

    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        k: int = config.K_DEFAULT,
        filters: Filter = non_filter,
        concurrencies: Iterable[int] = config.NUM_CONCURRENCY,
        duration: int = config.CONCURRENCY_DURATION,
        concurrency_timeout: int = config.CONCURRENCY_TIMEOUT,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self.concurrencies = concurrencies
        self.duration = duration
        self.concurrency_timeout = concurrency_timeout

        self.test_data = test_data
        log.debug(f"test dataset columns: {len(test_data)}")

    def search(
        self,
        test_data: list[list[float]],
        q: mp.Queue,
        cond: mp.Condition,
    ) -> tuple[int, float]:
        # sync all process
        q.put(1)
        with cond:
            cond.wait()

        with self.db.init():
            self.db.prepare_filter(self.filters)
            num, idx = len(test_data), random.randint(0, len(test_data) - 1)

            start_time = time.perf_counter()
            count = 0
            latencies = []
            while time.perf_counter() < start_time + self.duration:
                s = time.perf_counter()
                try:
                    self.db.search_embedding(test_data[idx], self.k)
                    count += 1
                    latencies.append(time.perf_counter() - s)
                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")

                # loop through the test data
                idx = idx + 1 if idx < num - 1 else 0

                if count % 500 == 0:
                    log.debug(
                        f"({mp.current_process().name:16}) "
                        f"search_count: {count}, latest_latency={time.perf_counter()-s}"
                    )

        total_dur = round(time.perf_counter() - start_time, 4)
        log.info(
            f"{mp.current_process().name:16} search {self.duration}s: "
            f"actual_dur={total_dur}s, count={count}, qps in this process: {round(count / total_dur, 4):3}"
        )

        return (count, total_dur, latencies)

    @staticmethod
    def get_mp_context():
        mp_start_method = "spawn"
        log.debug(f"MultiProcessingSearchRunner get multiprocessing start method: {mp_start_method}")
        return mp.get_context(mp_start_method)

    def _run_all_concurrencies_mem_efficient(self):
        max_qps = 0
        conc_num_list = []
        conc_qps_list = []
        conc_latency_p99_list = []
        conc_latency_p95_list = []
        conc_latency_avg_list = []
        try:
            for conc in self.concurrencies:
                with mp.Manager() as m:
                    q, cond = m.Queue(), m.Condition()
                    with concurrent.futures.ProcessPoolExecutor(
                        mp_context=self.get_mp_context(),
                        max_workers=conc,
                    ) as executor:
                        log.info(f"Start search {self.duration}s in concurrency {conc}, filters: {self.filters}")
                        future_iter = [executor.submit(self.search, self.test_data, q, cond) for i in range(conc)]
                        # Sync all processes
                        self._wait_for_queue_fill(q, size=conc)

                        with cond:
                            cond.notify_all()
                            log.info(f"Syncing all process and start concurrency search, concurrency={conc}")

                        start = time.perf_counter()
                        all_count = sum([r.result()[0] for r in future_iter])
                        latencies = sum([r.result()[2] for r in future_iter], start=[])
                        latency_p99 = np.percentile(latencies, 99)
                        latency_p95 = np.percentile(latencies, 95)
                        latency_avg = np.mean(latencies)
                        cost = time.perf_counter() - start

                        qps = round(all_count / cost, 4)
                        conc_num_list.append(conc)
                        conc_qps_list.append(qps)
                        conc_latency_p99_list.append(latency_p99)
                        conc_latency_p95_list.append(latency_p95)
                        conc_latency_avg_list.append(latency_avg)
                        log.info(f"End search in concurrency {conc}: dur={cost}s, total_count={all_count}, qps={qps}")

                if qps > max_qps:
                    max_qps = qps
                    log.info(f"Update largest qps with concurrency {conc}: current max_qps={max_qps}")
        except Exception as e:
            log.warning(
                f"Fail to search, concurrencies: {self.concurrencies}, max_qps before failure={max_qps}, reason={e}"
            )
            traceback.print_exc()

            # No results available, raise exception
            if max_qps == 0.0:
                raise e from None

        finally:
            self.stop()

        return (
            max_qps,
            conc_num_list,
            conc_qps_list,
            conc_latency_p99_list,
            conc_latency_p95_list,
            conc_latency_avg_list,
        )

    def _wait_for_queue_fill(self, q: Queue, size: int):
        wait_t = 0
        while q.qsize() < size:
            sleep_t = size if size < 10 else 10
            wait_t += sleep_t
            if wait_t > self.concurrency_timeout > 0:
                raise ConcurrencySlotTimeoutError
            time.sleep(sleep_t)

    def run(self) -> float:
        """
        Returns:
            float: largest qps
        """
        return self._run_all_concurrencies_mem_efficient()

    def stop(self) -> None:
        pass

    def run_by_dur(self, duration: int) -> tuple[float, float]:
        """
        Returns:
            float: largest qps
            float: failed rate
        """
        return self._run_by_dur(duration)

    def _run_by_dur(self, duration: int) -> tuple[float, float]:
        """
        Returns:
            float: largest qps
            float: failed rate
        """
        max_qps = 0
        try:
            for conc in self.concurrencies:
                with mp.Manager() as m:
                    q, cond = m.Queue(), m.Condition()
                    with concurrent.futures.ProcessPoolExecutor(
                        mp_context=self.get_mp_context(),
                        max_workers=conc,
                    ) as executor:
                        log.info(f"Start search_by_dur {duration}s in concurrency {conc}, filters: {self.filters}")
                        future_iter = [
                            executor.submit(self.search_by_dur, duration, self.test_data, q, cond) for i in range(conc)
                        ]
                        # Sync all processes
                        while q.qsize() < conc:
                            sleep_t = conc if conc < 10 else 10
                            time.sleep(sleep_t)

                        with cond:
                            cond.notify_all()
                            log.info(f"Syncing all process and start concurrency search, concurrency={conc}")

                        start = time.perf_counter()
                        res = [r.result() for r in future_iter]
                        all_success_count = sum([r[0] for r in res])
                        all_failed_count = sum([r[1] for r in res])
                        failed_rate = all_failed_count / (all_failed_count + all_success_count)
                        cost = time.perf_counter() - start

                        qps = round(all_success_count / cost, 4)
                        log.info(
                            f"End search in concurrency {conc}: dur={cost}s, failed_rate={failed_rate}, "
                            f"all_success_count={all_success_count}, all_failed_count={all_failed_count}, qps={qps}",
                        )
                if qps > max_qps:
                    max_qps = qps
                    log.info(f"Update largest qps with concurrency {conc}: current max_qps={max_qps}")
        except Exception as e:
            log.warning(
                f"Fail to search all concurrencies: {self.concurrencies}, max_qps before failure={max_qps}, reason={e}",
            )
            traceback.print_exc()

            # No results available, raise exception
            if max_qps == 0.0:
                raise e from None

        finally:
            self.stop()

        return max_qps, failed_rate

    def search_by_dur(self, dur: int, test_data: list[list[float]], q: mp.Queue, cond: mp.Condition) -> tuple[int, int]:
        """
        Returns:
            int: successful requests count
            int: failed requests count
        """
        # sync all process
        q.put(1)
        with cond:
            cond.wait()

        with self.db.init():
            self.db.prepare_filter(self.filters)
            num, idx = len(test_data), random.randint(0, len(test_data) - 1)

            start_time = time.perf_counter()
            success_count = 0
            failed_cnt = 0
            while time.perf_counter() < start_time + dur:
                s = time.perf_counter()
                try:
                    self.db.search_embedding(test_data[idx], self.k)
                    success_count += 1
                except Exception as e:
                    failed_cnt += 1
                    # reduce log
                    if failed_cnt <= 3:
                        log.warning(f"VectorDB search_embedding error: {e}")
                    else:
                        log.debug(f"VectorDB search_embedding error: {e}")

                # loop through the test data
                idx = idx + 1 if idx < num - 1 else 0

                if success_count % 500 == 0:
                    log.debug(
                        f"({mp.current_process().name:16}) search_count: {success_count}, "
                        f"latest_latency={time.perf_counter()-s}",
                    )

        total_dur = round(time.perf_counter() - start_time, 4)
        log.debug(
            f"{mp.current_process().name:16} search {self.duration}s: "
            f"actual_dur={total_dur}s, count={success_count}, failed_cnt={failed_cnt}, "
            f"qps (successful) in this process: {round(success_count / total_dur, 4):3}",
        )

        return success_count, failed_cnt
