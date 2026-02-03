import concurrent
import logging
import math
import multiprocessing as mp
import time
import traceback

import numpy as np
import psutil

from vectordb_bench.backend.dataset import DatasetManager, FtsDatasetManager
from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

from ... import config
from ...metric import calc_mrr, calc_ndcg, calc_ndcg_fts, calc_recall, calc_recall_fts, get_ideal_dcg
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

    def retry_insert(self, db: api.VectorDB, retry_idx: int = 0, **kwargs):
        _, error = db.insert_embeddings(**kwargs)
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
        count, _ = self._insert_all_batches()
        return count


class SerialFtsInsertRunner:
    """Simple serial FTS document insertion runner, similar to SerialInsertRunner.

    Inserts FTS documents sequentially in batches without rate control.
    """

    def __init__(
        self,
        db: api.VectorDB,
        dataset: FtsDatasetManager,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db

    def retry_insert(self, db: api.VectorDB, texts: list[str], doc_ids: list[int], retry_idx: int = 0):
        """Retry FTS insert with exponential backoff."""
        _, error = db.insert_documents(texts=texts, doc_ids=doc_ids)
        if error is not None:
            log.warning(f"FTS Insert Failed, try_idx={retry_idx}, Exception: {error}")
            retry_idx += 1
            if retry_idx <= config.MAX_INSERT_RETRY:
                time.sleep(retry_idx)
                self.retry_insert(db, texts=texts, doc_ids=doc_ids, retry_idx=retry_idx)
            else:
                msg = f"FTS Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
                raise RuntimeError(msg) from None

    def task(self) -> int:
        """Insert all FTS documents sequentially with buffered batch insertion.

        Processes documents row by row, but accumulates them in a buffer
        and inserts in batches for better performance.
        """
        count = 0
        buffer_size = config.NUM_PER_BATCH  # Use same batch size as dataset iterator
        insert_buffer: list[tuple[int, str]] = []  # List of (doc_id, text) tuples

        def flush_buffer() -> int:
            """Insert all documents in buffer and return count."""
            if not insert_buffer:
                return 0

            doc_ids = [doc_id for doc_id, _ in insert_buffer]
            texts = [text for _, text in insert_buffer]

            insert_count, error = self.db.insert_documents(texts=texts, doc_ids=doc_ids)
            if error is not None:
                self.retry_insert(self.db, texts=texts, doc_ids=doc_ids)
                insert_count = len(doc_ids)  # Assume all inserted after retry success

            assert insert_count == len(doc_ids)
            insert_buffer.clear()
            return insert_count

        with self.db.init():
            log.info(f"Start inserting FTS documents with batch size {buffer_size}")
            start = time.perf_counter()

            # Process documents row by row
            for batch in self.dataset:
                for doc in batch:
                    # Extract doc_id and text from FtsDocument object
                    doc_id = doc.doc_id if hasattr(doc, "doc_id") else int(doc["doc_id"])
                    text = doc.text if hasattr(doc, "text") else doc["text"]

                    # Add to buffer
                    insert_buffer.append((doc_id, text))

                    # Flush buffer when it reaches batch size
                    if len(insert_buffer) >= buffer_size:
                        count += flush_buffer()
                        if count % 100_000 == 0:
                            log.info(f"Loaded {count} FTS documents into VectorDB")

            # Insert remaining documents in buffer
            if insert_buffer:
                count += flush_buffer()

            log.info(f"Finish loading all FTS dataset into VectorDB, dur={time.perf_counter() - start:.2f}s")
            return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Insert FTS documents directly in main process.

        Note: Cannot use ProcessPoolExecutor because ir_datasets objects
        contain lambda functions that cannot be pickled.
        """
        # Execute directly in main process to avoid pickle issues with ir_datasets
        return self.task()

    def run(self) -> int:
        """Run FTS insertion and return total count."""
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
        search_fulltext: bool | None = None,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self._use_fts_metrics = False

        # Auto-detect search type if not explicitly specified
        if search_fulltext is None:
            # Check if test_data contains strings (FTS) or numbers (vector)
            first_item = test_data[0] if test_data else None
            search_fulltext = bool(isinstance(first_item, str))

        if search_fulltext:
            self._search_func = self.db.search_documents
        else:
            self._search_func = self.db.search_embedding
        self._use_fts_metrics = bool(search_fulltext)

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth

    def _get_db_search_res(self, emb: list[float] | str, retry_idx: int = 0) -> list[int]:
        try:
            results = self._search_func(emb, self.k)
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
            ideal_dcg = None if self._use_fts_metrics else get_ideal_dcg(self.k)

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {len(ground_truth)}")

            latencies, recalls, ndcgs, mrrs = [], [], [], []
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
                    if self._use_fts_metrics:
                        recalls.append(calc_recall_fts(self.k, gt, results))
                        ndcgs.append(calc_ndcg_fts(self.k, gt, results))
                        mrrs.append(calc_mrr(gt, results))
                    else:
                        recalls.append(calc_recall(self.k, gt[: self.k], results))
                        ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
                        mrrs.append(calc_mrr(gt[: self.k], results))
                else:
                    recalls.append(0)
                    ndcgs.append(0)
                    mrrs.append(0)

                if len(latencies) % 100 == 0:
                    log.debug(
                        f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
                        f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
                    )

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        avg_ndcg = round(np.mean(ndcgs), 4)
        avg_mrr = round(np.mean(mrrs), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        p95 = round(np.percentile(latencies, 95), 4)
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

    def _run_in_subprocess(self) -> tuple[float, float, float, float, float]:
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
    def run_with_cost(self) -> tuple[tuple[float, float, float, float, float], float]:
        """
        Search all test data in serial.
        Returns:
            tuple[tuple[float, float, float, float, float], float]:
            (avg_recall, avg_ndcg, avg_mrr, p99_latency, p95_latency), cost
        """
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()
