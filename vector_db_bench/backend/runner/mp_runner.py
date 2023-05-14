import time
import math
import concurrent
import multiprocessing as mp
import logging
from typing import Iterable, Any
import pandas as pd
import numpy as np
from ..clients import api
from .. import utils

log = logging.getLogger(__name__)

NUM_PER_BATCH = 5000

class MultiProcessingInsertRunner:
    def __init__(self, db: api.VectorDB, train_df: pd.DataFrame):
        log.info(f"shape: {train_df.shape}")
        self.db = db
        self.sharded_df = utils.SharedDataFrame(train_df)

        self.num_batches = math.ceil(train_df.shape[0]/NUM_PER_BATCH)
        self.tasks = [(self.sharded_df, idx) for idx in range(self.num_batches)]

    def insert_data(self, args):
        self.db.init()

        sharded_df, batch_id = args
        batch = sharded_df.read()[batch_id*NUM_PER_BATCH: (batch_id+1)*NUM_PER_BATCH]

        metadata, embeddings = batch['id'].to_list(), batch['emb'].to_list()
        log.debug(f"({mp.current_process().name:14})Batch No.{batch_id:3}: Start inserting {batch.shape[0]} embeddings")

        insert_results = self.db.insert_embeddings(
            embeddings=embeddings,
            metadata=metadata,
        )

        assert len(insert_results) == batch.shape[0]
        log.debug(f"({mp.current_process().name:14})Batch No.{batch_id:3}: Finish inserting embeddings")

    def _insert_all_batches_sequentially(self) -> list[int]:
        results = []
        for t in self.tasks:
            results.append(self.insert_data(t))
        return results

    def _insert_all_batches(self) -> list[int]:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            future_iter = executor.map(self.insert_data, self.tasks)
            results = [r for r in future_iter]
        return results

    def run_sequentially_endlessness(self) -> int:
        """run forever"""
        count = 0
        start_time = time.perf_counter()
        try:
            while True:
                results = self._insert_all_batches_sequentially()
                count += len(results)
        except Exception as e:
            duration = time.perf_counter() - start_time
            log.info("load reach limit: dur={duration}, insertion counts={count}, err={str(e)}")
            return duration, count

    def run_sequentially(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches_sequentially()
        duration = time.time() - start_time
        log.info(f'Sequentially inserted {len(self.tasks)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results

    def run(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches()
        duration = time.time() - start_time
        log.info(f'multiprocessing inserted {len(self.tasks)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results

    def clean(self):
        self.sharded_df.unlink()


class MultiProcessingSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_df: pd.DataFrame,
        ground_truth: pd.DataFrame,
        k: int = 100,
        filters: Any | None = None,
        concurrencies: Iterable[int] = (1, 5, 10, 15, 20, 25, 30, 35),
        duration: int = 30,
    ):
        self.db = db
        self.shared_test = utils.SharedDataFrame(test_df)
        self.shared_ground_truth = utils.SharedDataFrame(ground_truth)
        self.k = k
        self.filters = filters
        self.concurrencies = concurrencies
        self.duration = duration


    def search(self, args: tuple[utils.SharedDataFrame, utils.SharedDataFrame]):
        self.db.init()
        self.db.ready_to_search()

        test_df, ground_truth = args[0].read(), args[1].read()

        num, idx = test_df.shape[0], 0
        log.debug(f"batch: {test_df}, batch shape: {test_df.shape}")

        start_time = time.perf_counter()
        latencies = []
        count = 0
        while time.perf_counter() < start_time + self.duration:
            s = time.perf_counter()
            try:
                results = self.db.search_embedding_with_score(
                    test_df['emb'][idx],
                    self.k,
                    self.filters,
                )
            except Exception as e:
                log.warn(str(e))
                return

            count += 1
            idx = idx + 1 if idx < num - 1 else 0 # loop through the embeddings
            dur = time.perf_counter() - s
            latencies.append(dur)
            log.debug(f"({mp.current_process().name:14}) serial latency: {dur}")

        logging.info(
            f"{mp.current_process().name:14} search {self.duration}s: "
            f"cost={np.sum(latencies):.4f}s, "
            f"queries={len(latencies)}, "
            f"avg_latency={round(np.mean(latencies), 4)}"
         )
        # TODO: calculate recall
        return (latencies, count)

    def _run_all_concurrencies(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=35) as executor:
            for conc in self.concurrencies:
                start = time.perf_counter()
                log.info(f"start search in concurrency {conc}")
                future_iter = executor.map(self.search, [(self.shared_test, self.shared_ground_truth) for i in range(conc)])

                all_latencies, all_count = [], 0
                for r in future_iter:
                    all_latencies.extend(r[0])
                    all_count += r[1]

                total = time.perf_counter() - start

                p99 = round(np.percentile(all_latencies, 99), 4)
                avg = round(np.mean(all_latencies), 4)
                qps = round(all_count / total, 4)
                log.info(f"end search in concurrency {conc}: dur={total}s, queries={len(all_latencies)}, qps={qps}, avg={avg}, p99={p99}")

    def _run_sequantially(self):
        log.info("start search sequentially")
        self.search((self.shared_test, self.shared_ground_truth))
        log.info("end search sequentially")

    def run(self, seq=False):
        if seq:
            self._run_sequantially()
        else:
            self._run_all_concurrencies()

    def clean(self):
        self.shared_test.unlink()
        self.shared_ground_truth.unlink()
