import time
import logging
import traceback
import concurrent
import multiprocessing as mp
import numpy as np
import pandas as pd

from ..clients import api
from .. import utils
from ...metric import calc_recall

log = logging.getLogger(__name__)

class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_df: pd.DataFrame,
        ground_truth: pd.DataFrame,
        k: int = 100,
        filters: dict | None = None,
    ):
        self.db = db
        self.k = k
        self.filters = filters

        self.shared_test = utils.SharedDataFrame(test_df)
        self.shared_ground_truth = utils.SharedDataFrame(ground_truth)


    def search(self, args: tuple[utils.SharedDataFrame, utils.SharedDataFrame]):
        with self.db.init():
            test_df, ground_truth = args[0].read(), args[1].read()

            num = test_df.shape[0]
            log.debug(f"test dataset columns: {test_df.columns}, shape: {test_df.shape}")
            log.debug(f"ground truth colums: {ground_truth.columns}, shape: {ground_truth.shape}")

            latencies, recalls = [], []
            for idx in range(num):
                s = time.perf_counter()
                try:
                    results = self.db.search_embedding_with_score(
                        test_df['emb'][idx],
                        self.k,
                        self.filters,
                    )

                except Exception as e:
                    log.warning(f"VectorDB search_embedding_with_score error: {e}")
                    traceback.print_exc(chain=True)
                    raise e from None

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

                if len(latencies) % 100 == 0:
                    log.debug(f"({mp.current_process().name:14}) search_count={len(latencies):3}, latest_latency={latencies[-1]}, latest recall={recalls[-1]}")

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        logging.info(
            f"{mp.current_process().name:14} search entire dataset: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}"
         )
        return (avg_recall, avg_latency, p99)


    def _run_in_subprocess(self) -> tuple[float, float, float]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.shared_test, self.shared_ground_truth))
            result = future.result()
            return result


    def run(self) -> tuple[float, float, float]:
        return self._run_in_subprocess()

    def stop(self) -> None:
        """stop to prevent resource leak"""
        if self.shared_test:
            self.shared_test.unlink()
        if self.shared_ground_truth:
            self.shared_ground_truth.unlink()
