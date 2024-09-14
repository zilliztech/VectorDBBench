import logging
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DataSetIterator
from vectordb_bench.backend.utils import time_it
from vectordb_bench import config

from .util import get_data, is_futures_completed, get_future_exceptions
log = logging.getLogger(__name__)


class RatedMultiThreadingInsertRunner:
    def __init__(
        self,
        rate: int, # numRows per second
        db: api.VectorDB,
        dataset_iter: DataSetIterator,
        normalize: bool = False,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, (int, float)) else None
        self.dataset = dataset_iter
        self.db = db
        self.normalize = normalize
        self.insert_rate = rate
        self.batch_rate = rate // config.NUM_PER_BATCH

    def send_insert_task(self, db, emb: list[list[float]], metadata: list[str]):
        db.insert_embeddings(emb, metadata)

    @time_it
    def run_with_rate(self, q: mp.Queue):
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            executing_futures = []

            @time_it
            def submit_by_rate() -> bool:
                rate = self.batch_rate
                for data in self.dataset:
                    emb, metadata = get_data(data, self.normalize)
                    executing_futures.append(executor.submit(self.send_insert_task, self.db, emb, metadata))
                    rate -= 1

                    if rate == 0:
                        return False
                return rate == self.batch_rate

            with self.db.init():
                while True:
                    start_time = time.perf_counter()
                    finished, elapsed_time = submit_by_rate()
                    if finished is True:
                        q.put(None, block=True)
                        log.info(f"End of dataset, left unfinished={len(executing_futures)}")
                        return

                    q.put(True, block=False)
                    wait_interval = 1 - elapsed_time if elapsed_time < 1 else 0.001

                    e, completed = is_futures_completed(executing_futures, wait_interval)
                    if completed is True:
                        ex = get_future_exceptions(executing_futures)
                        if ex is not None:
                            log.warn(f"task error, terminating, err={ex}")
                            q.put(None)
                            executor.shutdown(wait=True, cancel_futures=True)
                            raise ex
                        else:
                            log.debug(f"Finished {len(executing_futures)} insert-{config.NUM_PER_BATCH} task in 1s, wait_interval={wait_interval:.2f}")
                        executing_futures = []
                    else:
                        log.warning(f"Failed to finish tasks in 1s, {e}, waited={wait_interval:.2f}, try to check the next round")
                    dur = time.perf_counter() - start_time
                    if dur < 1:
                        time.sleep(1 - dur)
