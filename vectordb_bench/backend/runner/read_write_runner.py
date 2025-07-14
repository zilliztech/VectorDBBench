import concurrent
import concurrent.futures
import logging
import math
import multiprocessing as mp
import time
from collections.abc import Iterable

import numpy as np

from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, non_filter
from vectordb_bench.backend.utils import time_it
from vectordb_bench.metric import Metric

from .mp_runner import MultiProcessingSearchRunner
from .rate_runner import RatedMultiThreadingInsertRunner
from .serial_runner import SerialSearchRunner

log = logging.getLogger(__name__)


class ReadWriteRunner(MultiProcessingSearchRunner, RatedMultiThreadingInsertRunner):
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        insert_rate: int = 1000,
        normalize: bool = False,
        k: int = 100,
        filters: Filter = non_filter,
        concurrencies: Iterable[int] = (1, 15, 50),
        search_stages: Iterable[float] = (
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),  # search from insert portion, 0.0 means search from the start
        optimize_after_write: bool = True,
        read_dur_after_write: int = 300,  # seconds, search duration when insertion is done
        timeout: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size

        for stage in search_stages:
            assert 0.0 <= stage < 1.0, "each search stage should be in [0.0, 1.0)"
        self.search_stages = sorted(search_stages)
        self.optimize_after_write = optimize_after_write
        self.read_dur_after_write = read_dur_after_write

        log.info(
            f"Init runner, concurencys={concurrencies}, search_stages={self.search_stages}, "
            f"stage_search_dur={read_dur_after_write}",
        )

        if normalize:
            test_emb = np.array(dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            test_emb = test_emb.tolist()
        else:
            test_emb = dataset.test_data

        MultiProcessingSearchRunner.__init__(
            self,
            db=db,
            test_data=test_emb,
            k=k,
            filters=filters,
            concurrencies=concurrencies,
        )
        RatedMultiThreadingInsertRunner.__init__(
            self,
            rate=insert_rate,
            db=db,
            dataset_iter=iter(dataset),
            normalize=normalize,
        )
        self.serial_search_runner = SerialSearchRunner(
            db=db,
            test_data=test_emb,
            ground_truth=dataset.gt_data,
            k=k,
            filters=filters,
        )

    @time_it
    def run_optimize(self):
        """Optimize needs to run in differenct process for pymilvus schema recursion problem"""
        with self.db.init():
            log.info("Search after write - Optimize start")
            self.db.optimize(data_size=self.data_volume)
            log.info("Search after write - Optimize finished")

    def run_search(self, perc: int):
        log.info("Search after write - Serial search start")
        test_time = round(time.perf_counter(), 4)
        res, ssearch_dur = self.serial_search_runner.run()
        recall, ndcg, p99_latency, p95_latency = res
        log.info(
            f"Search after write - Serial search - recall={recall}, ndcg={ndcg}, "
            f"p99={p99_latency}, p95={p95_latency}, dur={ssearch_dur:.4f}",
        )
        log.info(
            f"Search after wirte - Conc search start, dur for each conc={self.read_dur_after_write}",
        )
        max_qps, conc_failed_rate = self.run_by_dur(self.read_dur_after_write)
        log.info(f"Search after wirte - Conc search finished, max_qps={max_qps}")

        return [(perc, test_time, max_qps, recall, ndcg, p99_latency, p95_latency, conc_failed_rate)]

    def run_read_write(self) -> Metric:
        """
        Test search performance with a fixed insert rate.
        - Insert requests are sent to VectorDB at a fixed rate within a dedicated insert process pool.
          - if the database cannot promptly process these requests, the process pool will accumulate insert tasks.
        - Search Tests are categorized into three types:
          - streaming_search: Initiates a new search test upon receiving a signal that the inserted data has
          reached the search_stage.
          - streaming_end_search: initiates a new search test after all data has been inserted.
          - optimized_search (optional): After the streaming_end_search, optimizes and initiates a search test.
        """
        m = Metric()
        with mp.Manager() as mp_manager:
            q = mp_manager.Queue()
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=2) as executor:
                insert_future = executor.submit(self.run_with_rate, q)
                streaming_search_future = executor.submit(self.run_search_by_sig, q)

                try:
                    start_time = time.perf_counter()
                    _, m.insert_duration = insert_future.result()
                    streaming_search_res = streaming_search_future.result()
                    if streaming_search_res is None:
                        streaming_search_res = []

                    streaming_end_search_future = executor.submit(self.run_search, 100)
                    streaming_end_search_res = streaming_end_search_future.result()

                    # Wait for read_write_futures finishing and do optimize and search
                    if self.optimize_after_write:
                        op_future = executor.submit(self.run_optimize)
                        _, m.optimize_duration = op_future.result()
                        log.info(f"Optimize cost {m.optimize_duration}s")
                        optimized_search_future = executor.submit(self.run_search, 110)
                        optimized_search_res = optimized_search_future.result()
                    else:
                        log.info("Skip optimization and search")
                        optimized_search_res = []

                    r = [*streaming_search_res, *streaming_end_search_res, *optimized_search_res]
                    m.st_search_stage_list = [d[0] for d in r]
                    m.st_search_time_list = [round(d[1] - start_time, 4) for d in r]
                    m.st_max_qps_list_list = [d[2] for d in r]
                    m.st_recall_list = [d[3] for d in r]
                    m.st_ndcg_list = [d[4] for d in r]
                    m.st_serial_latency_p99_list = [d[5] for d in r]
                    m.st_serial_latency_p95_list = [d[6] for d in r]
                    m.st_conc_failed_rate_list = [d[7] for d in r]

                except Exception as e:
                    log.warning(f"Read and write error: {e}")
                    executor.shutdown(wait=True, cancel_futures=True)
                    # raise e
        m.st_ideal_insert_duration = math.ceil(self.data_volume / self.insert_rate)
        log.info(f"Concurrent read write all done, results: {m}")
        return m

    def get_each_conc_search_dur(self, ssearch_dur: float, cur_stage: float, next_stage: float) -> float:
        # Search duration for non-last search stage is carefully calculated.
        # If duration for each concurrency is less than 30s, runner will raise error.
        total_dur_between_stages = self.data_volume * (next_stage - cur_stage) // self.insert_rate
        csearch_dur = total_dur_between_stages - ssearch_dur

        # Try to leave room for init process executors
        if csearch_dur > 60:
            csearch_dur -= 30
        elif csearch_dur > 30:
            csearch_dur -= 15
        else:
            csearch_dur /= 2

        each_conc_search_dur = round(csearch_dur / len(self.concurrencies), 4)
        if each_conc_search_dur < 30:
            warning_msg = (
                f"Results might be inaccurate, duration[{csearch_dur:.4f}] left for conc-search is too short, "
                f"total available dur={total_dur_between_stages}, serial_search_cost={ssearch_dur}, "
                f"each_conc_search_dur={each_conc_search_dur}."
            )
            log.warning(warning_msg)
        return each_conc_search_dur

    def run_search_by_sig(self, q: mp.Queue):
        """
        Args:
            q: multiprocessing queue
                (None) means abnormal exit
                (False) means updating progress
                (True) means normal exit
        """
        result, start_batch = [], 0
        total_batch = math.ceil(self.data_volume / self.insert_rate)
        recall, ndcg, p99_latency, p95_latency = None, None, None, None

        def wait_next_target(start: int, target_batch: int) -> bool:
            """Return False when receive True or None"""
            while start < target_batch:
                sig = q.get(block=True)

                if sig is None or sig is True:
                    return False
                start += 1
            return True

        for idx, stage in enumerate(self.search_stages):
            target_batch = int(total_batch * stage)
            perc = int(stage * 100)

            got = wait_next_target(start_batch, target_batch)
            if got is False:
                log.warning(f"Abnormal exit, target_batch={target_batch}, start_batch={start_batch}")
                return None

            log.info(f"Insert {perc}% done, total batch={total_batch}")
            test_time = round(time.perf_counter(), 4)
            max_qps, recall, ndcg, p99_latency, p95_latency, conc_failed_rate = 0, 0, 0, 0, 0, 0
            try:
                log.info(f"[{target_batch}/{total_batch}] Serial search - {perc}% start")
                res, ssearch_dur = self.serial_search_runner.run()
                ssearch_dur = round(ssearch_dur, 4)
                recall, ndcg, p99_latency, p95_latency = res
                log.info(
                    f"[{target_batch}/{total_batch}] Serial search - {perc}% done, "
                    f"recall={recall}, ndcg={ndcg}, p99={p99_latency}, p95={p95_latency}, dur={ssearch_dur}"
                )

                each_conc_search_dur = self.get_each_conc_search_dur(
                    ssearch_dur,
                    cur_stage=stage,
                    next_stage=self.search_stages[idx + 1] if idx < len(self.search_stages) - 1 else 1.0,
                )
                if each_conc_search_dur > 10:
                    log.info(
                        f"[{target_batch}/{total_batch}] Concurrent search - {perc}% start, "
                        f"dur={each_conc_search_dur:.4f}"
                    )
                    max_qps, conc_failed_rate = self.run_by_dur(each_conc_search_dur)
                else:
                    log.warning(f"Skip concurrent tests, each_conc_search_dur={each_conc_search_dur} less than 10s.")
            except Exception as e:
                log.warning(f"Streaming Search Failed at stage={stage}. Exception: {e}")
            result.append((perc, test_time, max_qps, recall, ndcg, p99_latency, p95_latency, conc_failed_rate))
            start_batch = target_batch

        # Drain the queue
        while q.empty() is False:
            q.get(block=True)
        return result
