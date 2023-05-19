import time
import traceback
import math
import concurrent
import multiprocessing as mp
import logging
from typing import Iterable
import pandas as pd
from ..clients import api
from .. import utils
from ... import NUM_PER_BATCH

log = logging.getLogger(__name__)

class MultiProcessingInsertRunner:
    def __init__(self, db: api.VectorDB, train_df: pd.DataFrame):
        log.info(f"Dataset shape: {train_df.shape}")
        self.db = db
        self.sharded_df = utils.SharedDataFrame(train_df)

        # seq
        self.seq_batches = math.ceil(train_df.shape[0]/NUM_PER_BATCH)

        # conc
        self.num_concurrency = mp.cpu_count()
        self.num_per_concurrency = train_df.shape[0] // self.num_concurrency
        self.conc_tasks = [(self.sharded_df, idx) for idx in range(self.num_concurrency)]


    def insert_data(self, args) -> int:
        with self.db.init():
            sharded_df, conc_id = args
            conc_data = sharded_df.read()[conc_id*self.num_per_concurrency: (conc_id+1)*self.num_per_concurrency]

            num_conc_batches = math.ceil(conc_data.shape[0]/NUM_PER_BATCH)
            log.info(f"({mp.current_process().name:16}) Conc {conc_id:3}, Start batch inserting {conc_data.shape[0]} embeddings")

            count = 0
            for batch_id in range(num_conc_batches):
                batch = conc_data[batch_id*NUM_PER_BATCH: (batch_id+1)*NUM_PER_BATCH]
                metadata, embeddings = batch['id'].to_list(), batch['emb'].to_list()

                log.debug(f"({mp.current_process().name:16}) Conc {conc_id:2}, batch {batch_id:3}, Start inserting {batch.shape[0]} embeddings")
                insert_results = self.db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                )

                assert len(insert_results) == batch.shape[0]
                count += len(insert_results)
                log.debug(f"({mp.current_process().name:16}) Conc {conc_id:2}, batch {batch_id:3}, Finish inserting {batch.shape[0]} embeddings")

        log.info(f"({mp.current_process().name:16}) Conc {conc_id:2}, Finish batch inserting {conc_data.shape[0]} embeddings")
        return count


    def load_data(self, args) -> int:
        count = self.insert_data(args)
        return count


    def _insert_all_batches_sequentially(self) -> int:
        """Load case only"""
        with self.db.init():
            count = 0

            for idx in range(self.seq_batches):
                batch = self.sharded_df.read()[idx*NUM_PER_BATCH: (idx+1)*NUM_PER_BATCH]
                metadata, embeddings = batch['id'].to_list(), batch['emb'].to_list()
                log.debug(f"({mp.current_process().name:14})Batch No.{idx:3}: Start inserting {batch.shape[0]} embeddings")

                insert_results = self.db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                )

                assert len(insert_results) == batch.shape[0]
                log.debug(f"({mp.current_process().name:14})Batch No.{idx:3}: Finish inserting embeddings")

                if idx == 0:
                    self.db.ready_to_load()
                count += len(insert_results)
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
        if self.sharded_df:
            self.sharded_df.unlink()


class MultiProcessingSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_df: pd.DataFrame,
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

        self.shared_test = utils.SharedDataFrame(test_df)
        log.debug(f"test dataset columns: {test_df.columns}, shape: {test_df.shape}")


    def search(self, test_df: utils.SharedDataFrame) -> tuple[int, float]:
        with self.db.init():
            test_df = test_df.read()

            num, idx = test_df.shape[0], 0
            start_time = time.perf_counter()
            count = 0
            while time.perf_counter() < start_time + self.duration:
                s = time.perf_counter()
                try:
                    self.db.search_embedding_with_score(
                        test_df['emb'][idx],
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


    def _run_all_concurrencies(self) -> float:
        max_qps = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=sum(self.concurrencies)) as executor:
        #  with concurrent.futures.ProcessPoolExecutor(max_workers=self.concurrencies[-1]) as executor:
            for conc in self.concurrencies:
                start = time.perf_counter()
                log.info(f"start search {self.duration}s in concurrency {conc}, filters: {self.filters}")

                future_iter = executor.map(self.search, [self.shared_test for i in range(conc)])

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
        return self._run_all_concurrencies()

    def stop(self) -> None:
        if self.shared_test:
            self.shared_test.unlink()
