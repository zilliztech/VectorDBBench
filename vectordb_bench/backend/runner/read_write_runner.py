import concurrent
import logging
import math
import multiprocessing as mp
from collections.abc import Iterable

import numpy as np

from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager

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
        filters: dict | None = None,
        concurrencies: Iterable[int] = (1, 15, 50),
        search_stage: Iterable[float] = (
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),  # search from insert portion, 0.0 means search from the start
        read_dur_after_write: int = 300,  # seconds, search duration when insertion is done
        timeout: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size

        for stage in search_stage:
            assert 0.0 <= stage < 1.0, "each search stage should be in [0.0, 1.0)"
        self.search_stage = sorted(search_stage)
        self.read_dur_after_write = read_dur_after_write

        log.info(
            f"Init runner, concurencys={concurrencies}, search_stage={search_stage}, "
            f"stage_search_dur={read_dur_after_write}"
        )

        test_emb = np.stack(dataset.test_data["emb"])
        if normalize:
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
        test_emb = test_emb.tolist()

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
        )

    def run_optimize(self):
        """Optimize needs to run in differenct process for pymilvus schema recursion problem"""
        with self.db.init():
            log.info("Search after write - Optimize start")
            self.db.optimize(data_size=self.data_volume)
            log.info("Search after write - Optimize finished")

    def run_search(self):
        log.info("Search after write - Serial search start")
        res, ssearch_dur = self.serial_search_runner.run()
        recall, ndcg, p99_latency = res
        log.info(
            f"Search after write - Serial search - recall={recall}, ndcg={ndcg}, p99={p99_latency}, "
            f"dur={ssearch_dur:.4f}",
        )
        log.info(f"Search after wirte - Conc search start, dur for each conc={self.read_dur_after_write}")
        max_qps = self.run_by_dur(self.read_dur_after_write)
        log.info(f"Search after wirte - Conc search finished, max_qps={max_qps}")

        return (max_qps, recall, ndcg, p99_latency)

    def run_read_write(self):
        with mp.Manager() as m:
            q = m.Queue()
            with concurrent.futures.ProcessPoolExecutor(
                mp_context=mp.get_context("spawn"),
                max_workers=2,
            ) as executor:
                read_write_futures = []
                read_write_futures.append(executor.submit(self.run_with_rate, q))
                read_write_futures.append(executor.submit(self.run_search_by_sig, q))

                try:
                    for f in concurrent.futures.as_completed(read_write_futures):
                        res = f.result()
                        log.info(f"Result = {res}")

                    # Wait for read_write_futures finishing and do optimize and search
                    op_future = executor.submit(self.run_optimize)
                    op_future.result()

                    search_future = executor.submit(self.run_search)
                    last_res = search_future.result()

                    log.info(f"Max QPS after optimze and search: {last_res}")
                except Exception as e:
                    log.warning(f"Read and write error: {e}")
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise e from e
        log.info("Concurrent read write all done")

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
        recall, ndcg, p99_latency = None, None, None

        def wait_next_target(start: int, target_batch: int) -> bool:
            """Return False when receive True or None"""
            while start < target_batch:
                sig = q.get(block=True)

                if sig is None or sig is True:
                    return False
                start += 1
            return True

        for idx, stage in enumerate(self.search_stage):
            target_batch = int(total_batch * stage)
            perc = int(stage * 100)

            got = wait_next_target(start_batch, target_batch)
            if got is False:
                log.warning(f"Abnormal exit, target_batch={target_batch}, start_batch={start_batch}")
                return None

            log.info(f"Insert {perc}% done, total batch={total_batch}")
            log.info(f"[{target_batch}/{total_batch}] Serial search - {perc}% start")
            res, ssearch_dur = self.serial_search_runner.run()
            recall, ndcg, p99_latency = res
            log.info(
                f"[{target_batch}/{total_batch}] Serial search - {perc}% done, recall={recall}, "
                f"ndcg={ndcg}, p99={p99_latency}, dur={ssearch_dur:.4f}"
            )

            # Search duration for non-last search stage is carefully calculated.
            # If duration for each concurrency is less than 30s, runner will raise error.
            if idx < len(self.search_stage) - 1:
                total_dur_between_stages = self.data_volume * (self.search_stage[idx + 1] - stage) // self.insert_rate
                csearch_dur = total_dur_between_stages - ssearch_dur

                # Try to leave room for init process executors
                csearch_dur = csearch_dur - 30 if csearch_dur > 60 else csearch_dur

                each_conc_search_dur = csearch_dur / len(self.concurrencies)
                if each_conc_search_dur < 30:
                    warning_msg = (
                        f"Results might be inaccurate, duration[{csearch_dur:.4f}] left for conc-search is too short, "
                        f"total available dur={total_dur_between_stages}, serial_search_cost={ssearch_dur}."
                    )
                    log.warning(warning_msg)

            # The last stage
            else:
                each_conc_search_dur = 60

            log.info(
                f"[{target_batch}/{total_batch}] Concurrent search - {perc}% start, dur={each_conc_search_dur:.4f}"
            )
            max_qps = self.run_by_dur(each_conc_search_dur)
            result.append((perc, max_qps, recall, ndcg, p99_latency))

            start_batch = target_batch

        # Drain the queue
        while q.empty() is False:
            q.get(block=True)
        return result
