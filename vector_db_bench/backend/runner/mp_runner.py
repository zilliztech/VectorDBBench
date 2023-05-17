import time
import math
import concurrent
import multiprocessing as mp
import logging
from typing import Iterable
import pandas as pd
import numpy as np
from ..clients import api
from .. import utils
from ...metric import calc_recall, Metric

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

    def run_sequentially_endlessness(self) -> Metric:
        """run forever util DB raises exception or crash"""
        m = Metric(load_time=0.0, max_load_count=0)
        start_time = time.perf_counter()
        try:
            while True:
                results = self._insert_all_batches_sequentially()
                m.max_load_count += len(results)
        except Exception as e:
            m.load_time = time.perf_counter() - start_time
            log.info("load reach limit: dur={duration}, insertion counts={count}, err={e}")

            return m

    def run_sequentially(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches_sequentially()
        duration = time.time() - start_time
        log.info(f'Sequentially inserted {len(self.tasks)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results

    def run(self) -> list[int]:
        log.info(f'start insert {len(self.tasks)} batches of {NUM_PER_BATCH} entities')
        start_time = time.time()
        results = self._insert_all_batches()
        duration = time.time() - start_time
        log.info(f'end insert {len(self.tasks)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results

    def stop(self) -> None:
        # TODO
        self.clean()

    def clean(self):
        if self.sharded_df:
            self.sharded_df.unlink()


class MultiProcessingSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_df: pd.DataFrame,
        ground_truth: pd.DataFrame,
        k: int = 100,
        filters: dict | None = None,
        #  concurrencies: Iterable[int] = (1, 5, 10, 15, 20, 25, 30, 35),
        concurrencies: Iterable[int] = (1,),
        duration: int = 30,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self.concurrencies = concurrencies
        self.duration = duration

        self.shared_test = utils.SharedDataFrame(test_df)
        self.shared_ground_truth = utils.SharedDataFrame(ground_truth)


    def search(self, args: tuple[utils.SharedDataFrame, utils.SharedDataFrame]):
        self.db.init()
        #  self.db.ready_to_search()

        test_df, ground_truth = args[0].read(), args[1].read()

        num, idx = test_df.shape[0], 0
        log.debug(f"batch: {test_df.columns}, batch shape: {test_df.shape}")
        log.debug(f"ground_truth: {ground_truth.columns}, "
            f"ground_truth shape: {test_df.shape}, "
            f"ground_truth neighbors: {len(ground_truth['neighbors_id'])}")

        start_time = time.perf_counter()
        latencies = []
        recalls = []
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
                log.warning(f"{str(e)}, {e}")
                return

            latencies.append(time.perf_counter() - s)

            gt = ground_truth['neighbors_id'][idx]

            # valid_idx for ground_truth for no filter and high filter
            valid_idx = self.k

            # calculate the ground_truth for low filter, filtering 100 entities.
            if self.filters and self.filters['id'] == 100:
                valid_idx, iter_idx = 0, 0
                while iter_idx < self.k and valid_idx < len(gt):
                    if gt[iter_idx] >= self.filters['id']:
                        valid_idx += 1
                    iter_idx += 1

            recalls.append(calc_recall(self.k, gt[:valid_idx], results))
            log.debug(f"({mp.current_process().name:14}) serial latency: {latencies[-1]}, recall: {recalls[-1]}")

            count += 1
            idx = idx + 1 if idx < num - 1 else 0 # loop through the embeddings

        logging.info(
            f"{mp.current_process().name:14} search {self.duration}s: "
            f"cost={np.sum(latencies):.4f}s, "
            f"queries={len(latencies)}, "
            f"avg_latency={round(np.mean(latencies), 4)}, "
            f"avg_recall={round(np.mean(recalls), 4)}"
         )
        return (latencies, count, np.mean(recalls, dtype=float))

    @utils.time_it
    def _ready_to_search(self):
        self.db.init()
        self.db.ready_to_search()

    def _run_all_concurrencies(self) -> Metric:
        m = Metric()
        with concurrent.futures.ProcessPoolExecutor(max_workers=35) as executor:
            future = executor.submit(self._ready_to_search)
            _, m.build_duration = future.result()

            for conc in self.concurrencies:
                start = time.perf_counter()
                log.info(f"start search in concurrency {conc}, filters: {self.filters}")

                future_iter = executor.map(self.search, [(self.shared_test, self.shared_ground_truth) for i in range(conc)])

                all_latencies, all_count, recalls = [], 0, []
                for r in future_iter:
                    all_latencies.extend(r[0])
                    all_count += r[1]
                    recalls.append(r[2])

                total = time.perf_counter() - start
                qps = round(all_count / total, 4)
                log.info(f"end search in concurrency {conc}: dur={total}s, queries={len(all_latencies)}, qps={qps}")

                if qps > m.qps:
                    m.qps = float(qps)
                    m.p99 = float(round(np.percentile(all_latencies, 99), 4))
                    m.serial_latency = float(round(np.mean(all_latencies), 4))
                    m.recall = float(round(np.mean(recalls), 4))

                    log.info(f"update largest qps with concurrency {conc}: "
                        f"dur={total}s, queries={len(all_latencies)}, "
                        f"metric={m.model_dump(include=['qps', 'p99', 'serial_latency', 'recall', 'build_duration'])}")
            return m


    def run(self) -> Metric:
        return self._run_all_concurrencies()

    def stop(self) -> None:
        self.clean()
        pass # TODO

    def clean(self):
        if self.shared_test:
            self.shared_test.unlink()
        if self.shared_ground_truth:
            self.shared_ground_truth.unlink()
