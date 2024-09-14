import logging
from typing import Iterable
import multiprocessing as mp
import concurrent
import numpy as np
import math

from .mp_runner import MultiProcessingSearchRunner
from .serial_runner import SerialSearchRunner
from .rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager

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
        search_stage: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), # search in any insert portion, 0.0 means search from the start
        read_dur_after_write: int = 300, # seconds, search duration when insertion is done
        timeout: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size

        for stage in search_stage:
            assert 0.0 <= stage <= 1.0, "each search stage should be in [0.0, 1.0]"
        self.search_stage = sorted(search_stage)
        self.read_dur_after_write = read_dur_after_write

        log.info(f"Init runner, concurencys={concurrencies}, search_stage={search_stage}, stage_search_dur={read_dur_after_write}")

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

    def run_read_write(self):
        futures = []
        with mp.Manager() as m:
            q = m.Queue()
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=2) as executor:
                futures.append(executor.submit(self.run_with_rate, q))
                futures.append(executor.submit(self.run_search_by_sig, q))

                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    log.info(f"Result = {res}")

        log.info("Concurrent read write all done")


    def run_search_by_sig(self, q):
        res = []
        total_batch = math.ceil(self.data_volume / self.insert_rate)
        batch = 0
        recall = 'x'

        for idx, stage in enumerate(self.search_stage):
            target_batch = int(total_batch * stage)
            while q.get(block=True):
                batch += 1
                if batch >= target_batch:
                    perc = int(stage * 100)
                    log.info(f"Insert {perc}% done, total batch={total_batch}")
                    log.info(f"[{batch}/{total_batch}] Serial search - {perc}% start")
                    recall, ndcg, p99 =self.serial_search_runner.run()

                    if idx < len(self.search_stage) - 1:
                        stage_search_dur = (self.data_volume  * (self.search_stage[idx + 1] - stage) // self.insert_rate) // len(self.concurrencies)
                        if stage_search_dur < 30:
                            log.warning(f"Search duration too short, please reduce concurrency count or insert rate, or increase dataset volume: dur={stage_search_dur}, concurrencies={len(self.concurrencies)}, insert_rate={self.insert_rate}")
                        log.info(f"[{batch}/{total_batch}] Conc search - {perc}% start, dur for each conc={stage_search_dur}s")
                    else:
                        last_search_dur = self.data_volume * (1.0 - stage) // self.insert_rate
                        stage_search_dur = last_search_dur + self.read_dur_after_write
                        log.info(f"[{batch}/{total_batch}] Last conc search - {perc}% start, [read_until_write|read_after_write|total] =[{last_search_dur}s|{self.read_dur_after_write}s|{stage_search_dur}s]")

                    max_qps = self.run_by_dur(stage_search_dur)
                    res.append((perc, max_qps, recall))
                    break
        return res
