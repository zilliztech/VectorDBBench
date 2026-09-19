import concurrent
import logging
import math
import multiprocessing as mp
import random
import time
import traceback
from collections.abc import Iterator

import numpy as np

from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, non_filter
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.workload import WorkloadKind

from ... import config
from ...metric import calc_mrr_fts, calc_ndcg_fts, calc_recall_fts, calc_vector_metrics
from ...models import LoadTimeoutError
from .. import utils
from ..clients import api

LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT
DEFAULT_INSERT_BATCH_SIZE = config.DEFAULT_INSERT_BATCH_SIZE

log = logging.getLogger(__name__)


class SerialInsertRunner:
    # FTS insert is intentionally not implemented here. FTS performance loading
    # goes through ConcurrentInsertRunner; serial FTS insert can be added later
    # if a capacity or serial-load FTS case needs it.
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        filters: Filter = non_filter,
        timeout: float | None = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
    ):
        if batch_size <= 0:
            msg = f"insert batch size must be greater than 0, got {batch_size}"
            raise ValueError(msg)
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.filters = filters
        self.batch_size = batch_size

    def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
        with self.db.init():
            # unique id for endlessness insertion
            all_metadata = [i + left_id for i in all_metadata]

            num_batches = math.ceil(len(all_embeddings) / self.batch_size)
            log.info(
                f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
                f"embeddings in batch {self.batch_size}"
            )
            count = 0
            for batch_id in range(num_batches):
                retry_count = 0
                already_insert_count = 0
                metadata = all_metadata[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]
                embeddings = all_embeddings[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

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
                f"batch {self.batch_size}"
            )
        return count

    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = next(self.dataset.iter_batches(self.batch_size))
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
        test_data: list,
        ground_truth: list[list[int]] | list[dict[str, int]],
        k: int = 100,
        filters: Filter = non_filter,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant_labels: list[str] | None = None,
        measure_recall: bool = True,
        workload_kind: WorkloadKind = WorkloadKind.VECTOR,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self.workload_kind = workload_kind
        self.payload_profile = payload_profile
        self.tenant_labels = tenant_labels or []
        self.measure_recall = measure_recall
        if workload_kind == WorkloadKind.FULL_TEXT:
            self._search_func = self.db.search_documents
            self._use_fts_metrics = True
        elif workload_kind == WorkloadKind.VECTOR:
            self._search_func = self._search_embedding
            self._use_fts_metrics = False
        else:
            msg = f"Unsupported search workload: {workload_kind}"
            raise NotImplementedError(msg)
        if self.workload_kind == WorkloadKind.VECTOR and not self.db.supports_payload_profile(self.payload_profile):
            msg = f"{self.db.name} does not support payload_profile={self.payload_profile.value}"
            raise NotImplementedError(msg)
        if self.workload_kind == WorkloadKind.FULL_TEXT and not self.db.supports_document_payload_profile(
            self.payload_profile
        ):
            msg = f"{self.db.name} does not support document payload_profile={self.payload_profile.value}"
            raise NotImplementedError(msg)

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth

    def _search_embedding(self, emb: list[float], tenant: str | None = None) -> list[int]:
        if tenant is None:
            if self.payload_profile == PayloadProfile.IDS_ONLY:
                return self.db.search_embedding(emb, self.k)
            return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile)
        if self.payload_profile == PayloadProfile.IDS_ONLY:
            return self.db.search_embedding(emb, self.k, tenant=tenant)
        return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile, tenant=tenant)

    def _get_db_search_res(
        self,
        query: list[float] | str,
        tenant: str | None = None,
        retry_idx: int = 0,
    ) -> list[int]:
        try:
            if self.workload_kind == WorkloadKind.FULL_TEXT:
                if self.payload_profile == PayloadProfile.IDS_ONLY:
                    results = self._search_func(query, self.k)
                else:
                    results = self._search_func(query, self.k, payload_profile=self.payload_profile)
            else:
                results = self._search_func(query, tenant=tenant)
        except Exception as e:
            log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(query=query, tenant=tenant, retry_idx=retry_idx + 1)

            msg = f"Serial search failed and retried more than {config.MAX_SEARCH_RETRY} times"
            raise RuntimeError(msg) from e

        return results

    def _validate_ground_truth(self, test_data: list, ground_truth: object | None) -> None:
        if not self.measure_recall or ground_truth is None:
            return
        if len(test_data) != len(ground_truth):
            msg = f"Search query count {len(test_data)} does not match ground truth row count {len(ground_truth)}"
            raise ValueError(msg)
        if self._use_fts_metrics:
            return

        source_width = getattr(ground_truth, "width", None)
        if source_width is not None:
            if source_width < self.k:
                msg = f"Ground truth width {source_width} is smaller than requested K={self.k}"
                raise ValueError(msg)
            return

        for row_idx, row in enumerate(ground_truth):
            if len(row) < self.k:
                msg = f"Ground truth width {len(row)} at row {row_idx} is smaller than requested K={self.k}"
                raise ValueError(msg)

    @staticmethod
    def _iter_ground_truth(ground_truth: object) -> Iterator:
        if hasattr(ground_truth, "iter_rows"):
            return ground_truth.iter_rows()
        return iter(ground_truth)

    def search(self, args: tuple[list, object]) -> tuple[float, ...]:
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        test_data, ground_truth = args
        self._validate_ground_truth(test_data, ground_truth)
        ground_truth_iter = self._iter_ground_truth(ground_truth) if ground_truth is not None else None

        with self.db.init():
            self.db.prepare_filter(self.filters)

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {len(ground_truth) if ground_truth is not None else 0}")

            latencies, recalls, ndcgs, mrrs = [], [], [], []
            recall_at_samples = {}
            tenant_rng = random.Random(0)
            for emb in test_data:
                tenant = (
                    self.tenant_labels[tenant_rng.randrange(len(self.tenant_labels))]
                    if self.workload_kind == WorkloadKind.VECTOR and self.tenant_labels
                    else None
                )
                s = time.perf_counter()
                try:
                    results = self._get_db_search_res(emb, tenant=tenant)
                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    raise e from None

                latencies.append(time.perf_counter() - s)

                if self.measure_recall and ground_truth is not None:
                    gt = next(ground_truth_iter)
                    if self._use_fts_metrics:
                        recalls.append(calc_recall_fts(self.k, gt, results))
                        ndcgs.append(calc_ndcg_fts(self.k, gt, results))
                        mrrs.append(calc_mrr_fts(self.k, gt, results))
                    else:
                        recall, ndcg, recall_at = calc_vector_metrics(self.k, gt, results)
                        recalls.append(recall)
                        ndcgs.append(ndcg)
                        for cutoff, value in recall_at.items():
                            recall_at_samples.setdefault(cutoff, []).append(value)
                else:
                    recalls.append(0)
                    ndcgs.append(0)
                    if self._use_fts_metrics:
                        mrrs.append(0)

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
        p50 = round(np.percentile(latencies, 50), 4)
        if self._use_fts_metrics:
            avg_mrr = round(np.mean(mrrs), 4)
            log.info(
                f"{mp.current_process().name:14} search entire test_data: "
                f"cost={cost}s, "
                f"queries={len(latencies)}, "
                f"avg_recall={avg_recall}, "
                f"avg_ndcg={avg_ndcg}, "
                f"avg_mrr={avg_mrr}, "
                f"avg_latency={avg_latency}, "
                f"p99={p99}, "
                f"p95={p95}"
            )
            return (avg_recall, avg_ndcg, avg_mrr, p99, p95)

        avg_recall_at = {cutoff: round(np.mean(values), 4) for cutoff, values in recall_at_samples.items()}
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_ndcg={avg_ndcg}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}, "
            f"p95={p95}, "
            f"p50={p50}, "
            f"recall_at={avg_recall_at}"
        )
        return (avg_recall, avg_ndcg, p99, p95, p50, avg_recall_at)

    def _run_in_subprocess(self) -> tuple[float, ...]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            return future.result()

    @utils.time_it
    def run(self) -> tuple[float, ...]:
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()

    @utils.time_it
    def run_with_cost(self) -> tuple[tuple[float, ...], float]:
        """
        Search all test data in serial.
        Returns:
            tuple[tuple[float, ...], float]: search metrics and cost
        """
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()
