import time
import concurrent
import multiprocessing as mp
import logging
from typing import Type, Iterable, Any
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np
from ..clients import api
from .. import utils

log = logging.getLogger(__name__)

NUM_PER_BATCH = 5000


class MultiProcessingInsertRunner:
    def __init__(self, db_class: Type[api.VectorDB], train_df: pd.DataFrame):
        self.db = db_class
        self.batches = np.array_split(train_df, train_df.shape[0]/NUM_PER_BATCH)
        log.debug(f"{self.batches[0]}")

        self.batch_ids = [i for i in range(len(self.batches))]

    def insert_data(self, batch_id: int, batch: pd.DataFrame):
        db = self.db()
        log.debug(f"({mp.current_process().name:14})Batch No.{batch_id:3}: Start inserting {batch.shape[0]} embeddings")
        embeddings, metadatas = self.get_embedding_with_meta(batch)

        insert_results = db.insert_embeddings(
            embeddings=embeddings,
            metadatas=None,
        )
        assert len(insert_results) == batch.shape[0]
        log.debug(f"({mp.current_process().name:14})Batch No.{batch_id:3}: Finish inserting embeddings")

    def get_embedding_with_meta(self, df_data: pd.DataFrame) -> (list[list[float]], list[dict]):
        # TODO
        metadata = []
        embeddings = []

        for col in df_data.columns:
            if col == 'emb': #Cohere
                embeddings = df_data[col].to_list()
            #  else:
            #      field = {col: df_data[col].to_list()}
            #      metadata.append(field)
        return embeddings, metadata

    def _insert_all_batches_sequentially(self) -> list[int]:
        results = []
        for i in self.batch_ids:
            results.append(self.insert_data(i, self.batches[i]))
        return results

    def _insert_all_batches(self) -> list[int]:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            future_iter = executor.map(self.insert_data, self.batch_ids, self.batches)
            results = [r for r in future_iter]
        return results

    def run_sequentially(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches_sequentially()
        duration = time.time() - start_time
        log.info(f'Sequentially inserted {len(self.batch_ids)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results

    def run(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches()
        duration = time.time() - start_time
        log.info(f'multiprocessing inserted {len(self.batch_ids)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results


class MultiProcessingSearchRunner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    db_class: Type[api.VectorDB]
    test_df: pd.DataFrame
    ground_truth: pd.DataFrame
    k: int = 100
    filters: Any | None = None
    concurrencies: Iterable[int] = (1,)
    #  concurrencies: Iterable[int] = (1, 5, 10, 15, 20, 25, 30, 35)
    duration: int = 30

    def search(self, batch: pd.DataFrame):
        # TODO db
        db = self.db_class()
        num, idx = batch.shape[0], 0
        log.debug(f"batch: {batch}, batch shape: {batch.shape}")

        start_time = time.perf_counter()
        serial_latency_timer = utils.Timer("serial_latency")
        latencies = []
        count = 0
        while time.perf_counter() < start_time + self.duration:
            serial_latency_timer.start()
            try:
                results = db.search_embedding_with_score(
                    batch['emb'][idx],
                    self.k,
                    self.filters,
                )
            except Exception as e:
                log.warn(str(e))
                return

            count += 1
            idx = idx + 1 if idx < num - 1 else 0 # loop through the embeddings
            latencies.append(serial_latency_timer.stop())

        total = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        avg = round(np.mean(latencies), 4)
        qps = round(count / total, 4)
            # TODO: calculate recall
        logging.info(f"search {self.duration}s in process{mp.current_process().name}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")

    def _run_all_concurrencies(self):
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=35)
        for conc in self.concurrencies:
            log.info(f"start search in concurrency: {conc}")
            futures = [executor.submit(self.search, self.test_df) for i in range(conc)]
            for f  in futures:
                f.result()
            log.info(f"end search in concurrency: {conc}")

    def run(self):
        self._run_all_concurrencies()
