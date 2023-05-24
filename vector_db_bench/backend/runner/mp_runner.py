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
    def __init__(self, db: api.VectorDB, train_emb: np.ndarray, train_id: np.ndarray):
        log.info(f"Dataset shape: {train_emb.shape[0]}")
        self.db = db
        self.shared_emb = utils.SharedNumpyArray(train_emb)
        self.train_id = train_id.tolist()

        # seq
        self.seq_batches = math.ceil(train_emb.shape[0]/NUM_PER_BATCH)

        # conc
        self.num_concurrency = mp.cpu_count()
        self.num_per_concurrency = train_emb.shape[0] // self.num_concurrency
        self.conc_tasks = [(self.shared_emb, self.train_id, idx) for idx in range(self.num_concurrency)]


    def insert_data(self, args) -> int:
        with self.db.init():
            shared_np, train_id, conc_id = args
            all_embeddings = shared_np.read()[conc_id*self.num_per_concurrency: (conc_id+1)*self.num_per_concurrency].tolist()
            all_metadata = train_id[conc_id*self.num_per_concurrency: (conc_id+1)*self.num_per_concurrency]

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


    def load_data(self, args) -> int:
        count = self.insert_data(args)
        return count


    def _insert_all_batches_sequentially(self) -> int:
        """Load case only"""
        with self.db.init():
            all_embeddings = self.shared_emb.read().tolist()
            all_metadata = self.train_id
            count = 0

            for idx in range(self.seq_batches):
                metadata = all_metadata[idx*NUM_PER_BATCH: (idx+1)*NUM_PER_BATCH]
                embeddings = all_embeddings[idx*NUM_PER_BATCH: (idx+1)*NUM_PER_BATCH]
                log.debug(f"({mp.current_process().name:14})Batch No.{idx:3}: Start inserting {len(metadata)} embeddings")

                insert_count = self.db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                )

                assert insert_count == len(metadata)
                log.debug(f"({mp.current_process().name:14})Batch No.{idx:3}: Finish inserting embeddings")

                if idx == 0:
                    self.db.ready_to_load()
                count += insert_count
        return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_concurrency) as executor:
            future_iter = executor.map(self.insert_data, self.conc_tasks)
            total_count = sum([r for r in future_iter])
            return total_count

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
            import traceback
            log.info(f"load reach limit, insertion counts={utils.numerize(max_load_count)}, {max_load_count}, err={e}")
            traceback.print_exc()
            return max_load_count

    def run_sequentially(self) -> int:
        log.info(f'start sequentially insert {self.seq_batches} batches of {NUM_PER_BATCH} entities')
        count = self._insert_all_batches_sequentially()
        log.info(f'end sequentially insert {self.seq_batches} batches of {NUM_PER_BATCH} entities')
        return count

    def run(self) -> list[int]:
        log.info(f'start insert with concurrency of {self.num_concurrency}')
        count, dur = self._insert_all_batches()
        log.info(f'end insert with concurrency of {self.num_concurrency} in {dur} seconds')
        return count

    def stop(self) -> None:
        if self.shared_emb:
            self.shared_emb.unlink()


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
                    self.db.search_embedding_with_score(
                        test_data[idx],
                        self.k,
                        self.filters,
                    )
                except Exception as e:
                    log.warning(f"VectorDB search_embedding_with_score error: {e}")
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

    def _run_all_concurrencies_mem_efficient(self) -> float:
        max_qps = 0
        for conc in self.concurrencies:
            with concurrent.futures.ProcessPoolExecutor(max_workers=conc) as executor:
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

            max_qps = qps if qps > max_qps else max_qps
            log.info(f"update largest qps with concurrency {conc}: current max_qps={max_qps}")
        return max_qps


    def _run_all_concurrencies(self) -> float:
        max_qps = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=sum(self.concurrencies)) as executor:
            for conc in self.concurrencies:
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

                max_qps = qps if qps > max_qps else max_qps
                log.info(f"update largest qps with concurrency {conc}: current max_qps={max_qps}")
        return max_qps


    def run(self) -> float:
        """
        Returns:
            float: largest qps
        """
        #  return self._run_all_concurrencies()
        return self._run_all_concurrencies_mem_efficient()

    def stop(self) -> None:
        if self.test_data:
            self.test_data.unlink()
