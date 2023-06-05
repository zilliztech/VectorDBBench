import time
import traceback
import math
import concurrent
import multiprocessing as mp
import logging
from typing import Iterable
import numpy as np
from ..clients import api
from .. import utils
from ... import NUM_PER_BATCH

log = logging.getLogger(__name__)


class MultiProcessingInsertRunner:
    def __init__(self, db: api.VectorDB, train_emb: list[list[float]], train_id: list[int]):
        log.debug(f"Dataset shape: {len(train_emb)}")
        self.db = db
        self.shared_emb = train_emb
        self.train_id = train_id

        # seq
        self.seq_batches = math.ceil(len(train_emb)/NUM_PER_BATCH)

        # conc
        self.num_concurrency = 1


    def insert_data(self, args) -> int:
        with self.db.init():
            conc_id = args
            all_embeddings = self.shared_emb
            all_metadata = self.train_id

            num_conc_batches = math.ceil(len(all_embeddings)/NUM_PER_BATCH)
            log.info(f"({mp.current_process().name:16}) Conc {conc_id:3}, Start batch inserting {len(all_embeddings)} embeddings")
            count = 0
            for batch_id in range(num_conc_batches):
                metadata = all_metadata[batch_id*NUM_PER_BATCH: (batch_id+1)*NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id*NUM_PER_BATCH: (batch_id+1)*NUM_PER_BATCH]
                log.debug(f"({mp.current_process().name:16}) Conc {conc_id:2}, batch {batch_id:3}, Start inserting {len(metadata)} embeddings")
                insert_count = self.db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                )

                assert insert_count == len(metadata)
                count += insert_count
                log.debug(f"({mp.current_process().name:16}) Conc {conc_id:2}, batch {batch_id:3}, Finish inserting {len(metadata)} embeddings")

        log.info(f"({mp.current_process().name:16}) Conc {conc_id:2}, Finish batch inserting {len(all_embeddings)} embeddings")
        return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('spawn'), max_workers=self.num_concurrency) as executor:
            future_iter = executor.map(self.insert_data, (1,))
            total_count = sum([r for r in future_iter])
            return total_count


    def _insert_all_batches_sequentially(self) -> int:
        """Load case only"""
        with self.db.init():
            self.db.ready_to_load()
            count = 0

            for idx in range(self.seq_batches):
                metadata = self.train_id[idx*NUM_PER_BATCH: (idx+1)*NUM_PER_BATCH]
                embeddings = self.shared_emb[idx*NUM_PER_BATCH: (idx+1)*NUM_PER_BATCH]
                log.debug(f"({mp.current_process().name:14})Batch No.{idx:3}: Start inserting {len(metadata)} embeddings")

                insert_count = self.db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                )

                assert insert_count == len(metadata)
                count += insert_count
                log.debug(f"({mp.current_process().name:14})Batch No.{idx:3}: Finish inserting embeddings")
            return count


    def run_sequentially_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        max_load_count, times = 0, 0
        try:
            while True:
                count = self._insert_all_batches_sequentially()
                max_load_count += count
                times += 1
                log.info(f"Loaded {times:3} entire dataset, current max load counts={utils.numerize(max_load_count)}, {max_load_count}")
        except Exception as e:
            log.info(f"load reach limit, insertion counts={utils.numerize(max_load_count)}, {max_load_count}, err={e}")
            traceback.print_exc()
            return max_load_count

    def run_sequentially(self) -> int:
        log.info(f'start sequentially insert {self.seq_batches} batches of {NUM_PER_BATCH} entities')
        count = self._insert_all_batches_sequentially()
        log.info(f'end sequentially insert {self.seq_batches} batches of {NUM_PER_BATCH} entities')
        return count

    def run(self) -> int:
        log.info(f'start insert with concurrency of {self.num_concurrency}')
        count, dur = self._insert_all_batches()
        log.info(f'end insert with concurrency of {self.num_concurrency} in {dur} seconds')
        return count

    def stop(self) -> None:
        pass


class MultiProcessingSearchRunner:
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


    def search(self, test_np: utils.SharedNumpyArray) -> tuple[int, float]:
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

                if count % 50 == 0:
                    log.debug(f"({mp.current_process().name:16}) search_count: {count}, latest_latency={time.perf_counter()-s}")

        total_dur = round(time.perf_counter() - start_time, 4)
        logging.info(
            f"{mp.current_process().name:16} search {self.duration}s: "
            f"actual_dur={total_dur}s, count={count}, qps in this process: {round(count / total_dur, 4):3}"
         )

        return (count, total_dur)

    @staticmethod
    def get_mp_context():
        mp_start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        log.debug(f"MultiProcessingSearchRunner get multiprocessing start method: {mp_start_method}")
        return mp.get_context(mp_start_method)

    def _run_all_concurrencies_mem_efficient(self) -> float:
        max_qps = 0
        try:
            for conc in self.concurrencies:
                with concurrent.futures.ProcessPoolExecutor(mp_context=self.get_mp_context(), max_workers=conc) as executor:
                    start = time.perf_counter()
                    log.info(f"start search {self.duration}s in concurrency {conc}, filters: {self.filters}")
                    future_iter = executor.map(self.search, [self.test_data for i in range(conc)])

                    all_count = 0
                    for r in future_iter:
                        count, dur = r
                        all_count += count

                    cost = time.perf_counter() - start
                    qps = round(all_count / cost, 4)
                    log.info(f"end search in concurrency {conc}: dur={cost}s, total_count={all_count}, qps={qps}")

                if qps > max_qps:
                    max_qps = qps
                    log.info(f"update largest qps with concurrency {conc}: current max_qps={max_qps}")
        except Exception as e:
            log.warning(f"fail to search all concurrencies: {self.concurrencies}, max_qps before failure={max_qps}, reason={e}")
            traceback.print_exc()
            # No results available, raise exception
            if max_qps == 0:
                raise e from None

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
