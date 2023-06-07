import time
import logging
import traceback
import concurrent
import multiprocessing as mp
import math
import numpy as np
import pandas as pd

from ..clients import api
from ...metric import calc_recall
from .. import utils
from ... import config

NUM_PER_BATCH = config.NUM_PER_BATCH

log = logging.getLogger(__name__)


class SerialInsertRunner:
    def __init__(self, db: api.VectorDB, train_emb: list[list[float]], train_id: list[int]):
        log.debug(f"Dataset shape: {len(train_emb)}")
        self.db = db
        self.shared_emb = train_emb
        self.train_id = train_id

        self.seq_batches = math.ceil(len(train_emb)/NUM_PER_BATCH)

    def insert_data(self) -> int:
        with self.db.init():
            all_embeddings = self.shared_emb
            all_metadata = self.train_id

            num_conc_batches = math.ceil(len(all_embeddings)/NUM_PER_BATCH)
            log.info(f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} embeddings in batch {NUM_PER_BATCH}")
            count = 0
            for batch_id in range(self.seq_batches):
                metadata = all_metadata[batch_id*NUM_PER_BATCH: (batch_id+1)*NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id*NUM_PER_BATCH: (batch_id+1)*NUM_PER_BATCH]

                log.debug(f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_conc_batches}], Start inserting {len(metadata)} embeddings")
                insert_count = self.db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                )
                log.debug(f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_conc_batches}], Finish inserting {len(metadata)} embeddings")

                assert insert_count == len(metadata)
                count += insert_count
            log.info(f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in batch {NUM_PER_BATCH}")
        return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('spawn'), max_workers=1) as executor:
            future = executor.submit(self.insert_data)
            count = future.result()
            return count


    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        max_load_count, times = 0, 0
        try:
            with self.db.init():
                self.db.ready_to_load()
            while True:
                count = self._insert_data()
                max_load_count += count
                times += 1
                log.info(f"Loaded {times:3} entire dataset, current max load counts={utils.numerize(max_load_count)}, {max_load_count}")
        except Exception as e:
            log.info(f"load reach limit, insertion counts={utils.numerize(max_load_count)}, {max_load_count}, err={e}")
            traceback.print_exc()
            return max_load_count

    def run(self) -> int:
        count, dur = self._insert_all_batches()
        return count


class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        ground_truth: pd.DataFrame,
        k: int = 100,
        filters: dict | None = None,
    ):
        self.db = db
        self.k = k
        self.filters = filters

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth

    def search(self, args: tuple[list, pd.DataFrame]):
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        with self.db.init():
            test_data, ground_truth = args

            log.debug(f"test dataset size: {len(test_data)}")
            log.info(f"ground truth size: {ground_truth.columns}, shape: {ground_truth.shape}")

            latencies, recalls = [], []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                try:
                    results = self.db.search_embedding(
                        emb,
                        self.k,
                        self.filters,
                    )

                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    traceback.print_exc(chain=True)
                    raise e from None

                latencies.append(time.perf_counter() - s)

                gt = ground_truth['neighbors_id'][idx]
                #  gt = ground_truth['neighobrs_id'][idx]

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

                if len(latencies) % 100 == 0:
                    log.debug(f"({mp.current_process().name:14}) search_count={len(latencies):3}, latest_latency={latencies[-1]}, latest recall={recalls[-1]}")

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}"
         )
        return (avg_recall, p99)


    def _run_in_subprocess(self) -> tuple[float, float]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            result = future.result()
            return result

    def run(self) -> tuple[float, float]:
        return self._run_in_subprocess()
