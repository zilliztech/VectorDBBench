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

    def search(self, args: tuple[list, utils.SharedDataFrame]):
        with self.db.init():
            test_data, ground_truth = args

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {ground_truth.columns}, shape: {ground_truth.shape}")

            latencies, recalls = [], []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                try:
                    results = self.db.search_embedding_with_score(
                        emb,
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
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            result = future.result()
            return result


    def run(self) -> tuple[float, float, float]:
        return self._run_in_subprocess()

    def stop(self) -> None:
        """stop to prevent resource leak"""
        pass
