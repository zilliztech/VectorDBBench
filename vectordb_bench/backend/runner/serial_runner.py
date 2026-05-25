import concurrent.futures
import logging
import math
import multiprocessing as mp
import time
import traceback

import numpy as np

from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, non_filter

from ... import config
from ...metric import calc_ndcg, calc_recall, get_ideal_dcg
from ...models import LoadTimeoutError
from .. import utils
from ..clients import api

NUM_PER_BATCH = config.NUM_PER_BATCH
LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT

log = logging.getLogger(__name__)


class SerialInsertRunner:
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        filters: Filter = non_filter,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.filters = filters

    def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
        with self.db.init():
            # unique id for endlessness insertion
            all_metadata = [i + left_id for i in all_metadata]

            num_batches = math.ceil(len(all_embeddings) / NUM_PER_BATCH)
            log.info(
                f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
                f"embeddings in batch {NUM_PER_BATCH}"
            )
            count = 0
            for batch_id in range(num_batches):
                retry_count = 0
                already_insert_count = 0
                metadata = all_metadata[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]

                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Start inserting {len(metadata)} embeddings"
                )
                while retry_count < LOAD_MAX_TRY_COUNT:
                    insert_count, error = self.db.insert_embeddings(
                        embeddings=embeddings[already_insert_count:],
                        metadata=metadata[already_insert_count:],
                    )
                    already_insert_count += insert_count
                    if error is not None:
                        retry_count += 1
                        time.sleep(10)

                        log.info(f"Failed to insert data, try {retry_count} time")
                        if retry_count >= LOAD_MAX_TRY_COUNT:
                            raise error
                    else:
                        break
                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Finish inserting {len(metadata)} embeddings"
                )

                assert already_insert_count == len(metadata)
                count += already_insert_count
            log.info(
                f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in "
                f"batch {NUM_PER_BATCH}"
            )
        return count

    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = next(iter(self.dataset))
        all_embeddings, all_metadata = (
            np.stack(data_df[self.dataset.data.train_vector_field]).tolist(),
            data_df[self.dataset.data.train_id_field].tolist(),
        )

        start_time = time.perf_counter()
        max_load_count, times = 0, 0
        try:
            while time.perf_counter() - start_time < self.timeout:
                count = self.endless_insert_data(
                    all_embeddings,
                    all_metadata,
                    left_id=max_load_count,
                )
                max_load_count += count
                times += 1
                log.info(
                    f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, "
                    f"{max_load_count}"
                )
        except Exception as e:
            log.info(
                f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, "
                f"{max_load_count}, err={e}"
            )
            traceback.print_exc()
            return max_load_count
        else:
            raise LoadTimeoutError(self.timeout)


class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        ground_truth: list[list[int]],
        k: int = 100,
        filters: Filter = non_filter,
    ):
        self.db = db
        self.k = k
        self.filters = filters

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth

    def _get_db_search_res(self, emb: list[float], retry_idx: int = 0) -> list[int]:
        try:
            results = self.db.search_embedding(emb, self.k)
        except Exception as e:
            log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1)

            msg = f"Serial search failed and retried more than {config.MAX_SEARCH_RETRY} times"
            raise RuntimeError(msg) from e

        return results

    def search(self, args: tuple[list, list[list[int]]]) -> tuple[float, float, float, float, float]:
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        with self.db.init():
            self.db.prepare_filter(self.filters)
            test_data, ground_truth = args
            ideal_dcg = get_ideal_dcg(self.k)

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {len(ground_truth)}")

            latencies, recalls, ndcgs = [], [], []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                try:
                    results = self._get_db_search_res(emb)
                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    raise e from None

                latencies.append(time.perf_counter() - s)

                if ground_truth is not None:
                    gt = ground_truth[idx]
                    recalls.append(calc_recall(self.k, gt[: self.k], results))
                    ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
                else:
                    recalls.append(0)
                    ndcgs.append(0)

                if len(latencies) % 100 == 0:
                    log.debug(
                        f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
                        f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
                    )

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        avg_ndcg = round(np.mean(ndcgs), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        p95 = round(np.percentile(latencies, 95), 4)
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_ndcg={avg_ndcg}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}, "
            f"p95={p95}"
        )
        return (avg_recall, avg_ndcg, p99, p95, avg_latency)

    def _run_in_subprocess(self) -> tuple[float, float, float, float]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            return future.result()

    @utils.time_it
    def run(self) -> tuple[float, float, float, float, float]:
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()

    @utils.time_it
    def run_with_cost(self) -> tuple[tuple[float, float, float, float], float]:
        """
        Search all test data in serial.
        Returns:
            tuple[tuple[float, float, float, float], float]: (avg_recall, avg_ndcg, p99_latency, p95_latency), cost
        """
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()
