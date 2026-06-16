import concurrent
import logging
import math
import multiprocessing as mp
import random
import time
import traceback

import numpy as np
import psutil

from vectordb_bench.backend.dataset import DatasetManager, FtsDatasetManager
from vectordb_bench.backend.filter import Filter, FilterOp, non_filter
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.workload import WorkloadKind

from ... import config
from ...metric import calc_ndcg, calc_recall, calc_recall_fts, get_ideal_dcg
from ...models import LoadTimeoutError, PerformanceTimeoutError
from .. import utils
from ..clients import api

NUM_PER_BATCH = config.NUM_PER_BATCH
LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT

log = logging.getLogger(__name__)


class SerialInsertRunner:
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager | FtsDatasetManager,
        normalize: bool = False,
        filters: Filter = non_filter,
        timeout: float | None = None,
        workload_kind: WorkloadKind = WorkloadKind.VECTOR,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.filters = filters
        self.workload_kind = workload_kind

    def _insert_batch(self, db: api.VectorDB, **kwargs):
        if self.workload_kind == WorkloadKind.FULL_TEXT:
            return db.insert_documents(**kwargs)
        return db.insert_embeddings(**kwargs)

    def retry_insert(self, db: api.VectorDB, retry_idx: int = 0, **kwargs):
        _, error = self._insert_batch(db, **kwargs)
        if error is not None:
            log.warning(f"Insert Failed, try_idx={retry_idx}, Exception: {error}")
            retry_idx += 1
            if retry_idx <= config.MAX_INSERT_RETRY:
                time.sleep(retry_idx)
                self.retry_insert(db, retry_idx=retry_idx, **kwargs)
            else:
                msg = f"Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
                raise RuntimeError(msg) from None

    def task(self) -> int:
        if self.workload_kind == WorkloadKind.FULL_TEXT:
            return self._task_documents()
        return self._task_embeddings()

    def _task_embeddings(self) -> int:
        count = 0
        with self.db.init():
            log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
            start = time.perf_counter()
            for data_df in self.dataset:
                all_metadata = data_df[self.dataset.data.train_id_field].tolist()

                emb_np = np.stack(data_df[self.dataset.data.train_vector_field])
                if self.normalize:
                    log.debug("normalize the 100k train data")
                    all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
                else:
                    all_embeddings = emb_np.tolist()
                del emb_np
                log.debug(f"batch dataset size: {len(all_embeddings)}, {len(all_metadata)}")

                labels_data = None
                if self.filters.type == FilterOp.StrEqual:
                    if self.dataset.data.scalar_labels_file_separated:
                        labels_data = self.dataset.scalar_labels[self.filters.label_field][all_metadata].to_list()
                    else:
                        labels_data = data_df[self.filters.label_field].tolist()

                insert_count, error = self.db.insert_embeddings(
                    embeddings=all_embeddings,
                    metadata=all_metadata,
                    labels_data=labels_data,
                )
                if error is not None:
                    self.retry_insert(
                        self.db,
                        embeddings=all_embeddings,
                        metadata=all_metadata,
                        labels_data=labels_data,
                    )

                assert insert_count == len(all_metadata)
                count += insert_count
                if count % 100_000 == 0:
                    log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")

            log.info(
                f"({mp.current_process().name:16}) Finish loading all dataset into VectorDB, "
                f"dur={time.perf_counter() - start}"
            )
            return count

    def _task_documents(self) -> int:
        count = 0
        with self.db.init():
            log.info(f"({mp.current_process().name:16}) Start inserting FTS documents in batch {config.NUM_PER_BATCH}")
            start = time.perf_counter()
            for batch in self.dataset:
                doc_ids = []
                texts = []
                for doc in batch:
                    doc_ids.append(doc.doc_id if hasattr(doc, "doc_id") else str(doc["doc_id"]))
                    texts.append(doc.text if hasattr(doc, "text") else doc["text"])

                insert_count, error = self.db.insert_documents(texts=texts, doc_ids=doc_ids)
                if error is not None:
                    self.retry_insert(self.db, texts=texts, doc_ids=doc_ids)
                    insert_count = len(doc_ids)

                assert insert_count == len(doc_ids)
                count += insert_count
                if count % 100_000 == 0:
                    log.info(f"({mp.current_process().name:16}) Loaded {count} FTS documents into VectorDB")

            log.info(
                f"({mp.current_process().name:16}) Finish loading all FTS dataset into VectorDB, "
                f"dur={time.perf_counter() - start}"
            )
            return count

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

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        if self.workload_kind == WorkloadKind.FULL_TEXT:
            return self.task()
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("spawn"),
            max_workers=1,
        ) as executor:
            future = executor.submit(self.task)
            try:
                count = future.result(timeout=self.timeout)
            except TimeoutError as e:
                msg = f"VectorDB load dataset timeout in {self.timeout}"
                log.warning(msg)
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError(msg) from e
            except Exception as e:
                log.warning(f"VectorDB load dataset error: {e}")
                raise e from e
            else:
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

    def run(self) -> int:
        if self.workload_kind == WorkloadKind.FULL_TEXT:
            with utils.timeout(self.timeout, lambda: LoadTimeoutError(self.timeout)):
                count, _ = self._insert_all_batches()
            return count
        count, _ = self._insert_all_batches()
        return count

class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list,
        ground_truth: list[list[int]],
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
        if (
            self.workload_kind == WorkloadKind.FULL_TEXT
            and not self.db.supports_document_payload_profile(self.payload_profile)
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

    def search(self, args: tuple[list, list[list[int]]]) -> tuple[float, ...]:
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        with self.db.init():
            self.db.prepare_filter(self.filters)
            test_data, ground_truth = args
            ideal_dcg = None if self._use_fts_metrics else get_ideal_dcg(self.k)

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {len(ground_truth) if ground_truth is not None else 0}")

            latencies, recalls, ndcgs = [], [], []
            tenant_rng = random.Random(0)
            for idx, emb in enumerate(test_data):
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
                    gt = ground_truth[idx]
                    if self._use_fts_metrics:
                        recalls.append(calc_recall_fts(self.k, gt, results))
                    else:
                        recalls.append(calc_recall(self.k, gt[: self.k], results))
                        ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
                else:
                    recalls.append(0)
                    if not self._use_fts_metrics:
                        ndcgs.append(0)

                if len(latencies) % 100 == 0:
                    log.debug(
                        f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
                        f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
                    )

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        p95 = round(np.percentile(latencies, 95), 4)
        if self._use_fts_metrics:
            log.info(
                f"{mp.current_process().name:14} search entire test_data: "
                f"cost={cost}s, "
                f"queries={len(latencies)}, "
                f"avg_recall={avg_recall}, "
                f"avg_latency={avg_latency}, "
                f"p99={p99}, "
                f"p95={p95}"
            )
            return (avg_recall, p99, p95)

        avg_ndcg = round(np.mean(ndcgs), 4)
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
        return (avg_recall, avg_ndcg, p99, p95)

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
