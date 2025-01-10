import concurrent
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

from vectordb_bench import config
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DataSetIterator
from vectordb_bench.backend.utils import time_it

from .util import get_data

log = logging.getLogger(__name__)


class RatedMultiThreadingInsertRunner:
    def __init__(
        self,
        rate: int,  # numRows per second
        db: api.VectorDB,
        dataset_iter: DataSetIterator,
        normalize: bool = False,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset_iter
        self.db = db
        self.normalize = normalize
        self.insert_rate = rate
        self.batch_rate = rate // config.NUM_PER_BATCH

    def send_insert_task(self, db: api.VectorDB, emb: list[list[float]], metadata: list[str]):
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
                    executing_futures.append(
                        executor.submit(self.send_insert_task, self.db, emb, metadata),
                    )
                    rate -= 1

                    if rate == 0:
                        return False
                return rate == self.batch_rate

            with self.db.init():
                while True:
                    start_time = time.perf_counter()
                    finished, elapsed_time = submit_by_rate()
                    if finished is True:
                        q.put(True, block=True)
                        log.info(f"End of dataset, left unfinished={len(executing_futures)}")
                        break

                    q.put(False, block=False)
                    wait_interval = 1 - elapsed_time if elapsed_time < 1 else 0.001

                    try:
                        done, not_done = concurrent.futures.wait(
                            executing_futures,
                            timeout=wait_interval,
                            return_when=concurrent.futures.FIRST_EXCEPTION,
                        )

                        if len(not_done) > 0:
                            log.warning(
                                f"Failed to finish all tasks in 1s, [{len(not_done)}/{len(executing_futures)}] "
                                f"tasks are not done, waited={wait_interval:.2f}, trying to wait in the next round"
                            )
                            executing_futures = list(not_done)
                        else:
                            log.debug(
                                f"Finished {len(executing_futures)} insert-{config.NUM_PER_BATCH} "
                                f"task in 1s, wait_interval={wait_interval:.2f}"
                            )
                            executing_futures = []
                    except Exception as e:
                        log.warning(f"task error, terminating, err={e}")
                        q.put(None, block=True)
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise e from e

                    dur = time.perf_counter() - start_time
                    if dur < 1:
                        time.sleep(1 - dur)

                # wait for all tasks in executing_futures to complete
                if len(executing_futures) > 0:
                    try:
                        done, _ = concurrent.futures.wait(
                            executing_futures,
                            return_when=concurrent.futures.FIRST_EXCEPTION,
                        )
                    except Exception as e:
                        log.warning(f"task error, terminating, err={e}")
                        q.put(None, block=True)
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise e from e
