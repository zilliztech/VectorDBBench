import time
import traceback
import concurrent
import multiprocessing as mp
import logging
from typing import Iterable
import numpy as np
from ..clients import api
from .. import utils
from ... import config


NUM_PER_BATCH = config.NUM_PER_BATCH
log = logging.getLogger(__name__)


class MultiProcessingSearchRunner:
    """ multiprocessing search runner

    Args:
        k(int): search topk, default to 100
        concurrency(Iterable): concurrencies, default [1, 5, 10, 15, 20, 25, 30, 35]
        duration(int): duration for each concurency, default to 30s
    """
    def __init__(
        self,
        db: api.VectorDB,
        test_data: np.ndarray,
        k: int = 100,
        filters: dict | None = None,
        concurrencies: Iterable[int] = (1, 5, 10, 15, 20, 25, 30, 35),
        duration: int = 30,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self.concurrencies = concurrencies
        self.duration = duration

        self.test_data = utils.SharedNumpyArray(test_data)
        log.debug(f"test dataset columns: {len(test_data)}")

    def search(self, test_np: utils.SharedNumpyArray, q: mp.Queue, cond: mp.Condition) -> tuple[int, float]:
        # sync all process
        q.put(1)
        with cond:
            cond.wait()

        with self.db.init():
            test_data = test_np.read().tolist()
            num, idx = len(test_data), 0

            start_time = time.perf_counter()
            count = 0
            while time.perf_counter() < start_time + self.duration:
                s = time.perf_counter()
                try:
                    self.db.search_embedding(
                        test_data[idx],
                        self.k,
                        self.filters,
                    )
                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    traceback.print_exc(chain=True)
                    raise e from None

                count += 1
                # loop through the test data
                idx = idx + 1 if idx < num - 1 else 0

                if count % 500 == 0:
                    log.debug(f"({mp.current_process().name:16}) search_count: {count}, latest_latency={time.perf_counter()-s}")

        total_dur = round(time.perf_counter() - start_time, 4)
        log.info(
            f"{mp.current_process().name:16} search {self.duration}s: "
            f"actual_dur={total_dur}s, count={count}, qps in this process: {round(count / total_dur, 4):3}"
         )

        return (count, total_dur)

    @staticmethod
    def get_mp_context():
        mp_start_method = "spawn"
        log.debug(f"MultiProcessingSearchRunner get multiprocessing start method: {mp_start_method}")
        return mp.get_context(mp_start_method)

    def _run_all_concurrencies_mem_efficient(self) -> float:
        max_qps = 0
        try:
            for conc in self.concurrencies:
                with mp.Manager() as m:
                    q, cond = m.Queue(), m.Condition()
                    with concurrent.futures.ProcessPoolExecutor(mp_context=self.get_mp_context(), max_workers=conc) as executor:
                        log.info(f"Start search {self.duration}s in concurrency {conc}, filters: {self.filters}")
                        future_iter = [executor.submit(self.search, self.test_data, q, cond) for i in range(conc)]
                        # Sync all processes
                        while q.qsize() < conc:
                            sleep_t = conc if conc < 10 else 10
                            time.sleep(sleep_t)

                        with cond:
                            cond.notify_all()
                            log.info(f"Syncing all process and start concurrency search, concurrency={conc}")

                        start = time.perf_counter()
                        all_count = sum([r.result()[0] for r in future_iter])
                        cost = time.perf_counter() - start

                        qps = round(all_count / cost, 4)
                        log.info(f"End search in concurrency {conc}: dur={cost}s, total_count={all_count}, qps={qps}")

                if qps > max_qps:
                    max_qps = qps
                    log.info(f"Update largest qps with concurrency {conc}: current max_qps={max_qps}")
        except Exception as e:
            log.warning(f"Fail to search all concurrencies: {self.concurrencies}, max_qps before failure={max_qps}, reason={e}")
            traceback.print_exc()

            # No results available, raise exception
            if max_qps == 0.0:
                raise e from None

        finally:
            self.stop()

        return max_qps

    def run(self) -> float:
        """
        Returns:
            float: largest qps
        """
        return self._run_all_concurrencies_mem_efficient()

    def stop(self) -> None:
        if self.test_data:
            self.test_data.unlink()
            self.test_data = None
