"""Concurrent insert runner with configurable executor backend.

Replaces SerialInsertRunner for faster data loading in performance cases.

Auto-detects thread-unsafe DBs via VectorDB.thread_safe and
falls back to single-worker mode.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from vectordb_bench.backend.filter import Filter, FilterOp, non_filter
from vectordb_bench.backend.utils import kill_proc_tree, time_it, timeout as run_timeout

from ... import config
from ...models import LoadTimeoutError, PerformanceTimeoutError
from .executor import AsyncExecutor, ThreadExecutor

if TYPE_CHECKING:
    from vectordb_bench.backend.clients import api
    from vectordb_bench.backend.dataset import DatasetManager, FtsDatasetManager

    from .executor import TaskExecutor

log = logging.getLogger(__name__)


class ExecutorBackend(StrEnum):
    THREADING = "threading"
    ASYNC = "async"


class ConcurrentInsertRunner:
    """Concurrent insert runner with pluggable executor backend.

    Thread-safety: If db.thread_safe is False, max_workers is clamped to 1
    so the single worker thread uses self.db directly (no deepcopy needed).

    Args:
        db: VectorDB instance.
        dataset: DatasetManager for batch iteration.
        normalize: Whether to L2-normalize embeddings.
        filters: Filter configuration.
        timeout: Timeout in seconds for the overall operation.
        max_workers: Number of concurrent workers (default: min(cpu_count, 4)).
        backend: Executor backend to use ('threading' or 'async').
    """

    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        filters: Filter = non_filter,
        timeout: float | None = None,
        max_workers: int | None = None,
        backend: ExecutorBackend = ExecutorBackend.THREADING,
        batch_size: int = config.NUM_PER_BATCH,
        duration: float | None = None,
        with_scalar_labels: bool = False,
        tenant_case=None,  # noqa: ANN001
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset: DatasetManager = dataset
        self.db = db
        self.normalize = normalize
        self.filters = filters
        self.backend = backend
        self.batch_size = batch_size
        self.duration = duration if isinstance(duration, int | float) else None
        self.with_scalar_labels = with_scalar_labels
        self.tenant_case = tenant_case

        effective_workers = max_workers or min(mp.cpu_count(), 4)
        if not db.thread_safe:
            log.info(f"DB {db.name} is not thread-safe, falling back to max_workers=1")
            effective_workers = 1
        self.max_workers = effective_workers
        assert db.thread_safe or self.max_workers == 1, (
            "Non-thread-safe DBs must use max_workers=1 — "
            "_get_thread_db() relies on this to avoid concurrent access to self.db"
        )

    def __getstate__(self):
        """Exclude unpicklable thread-local state for ProcessPoolExecutor(spawn)."""
        state = self.__dict__.copy()
        state.pop("_iter_lock", None)
        state.pop("_dataset_iter", None)
        state.pop("_stop_event", None)
        return state

    def _create_executor(self) -> TaskExecutor:
        if self.backend == ExecutorBackend.ASYNC:
            return AsyncExecutor(max_workers=self.max_workers)
        return ThreadExecutor(max_workers=self.max_workers)

    def _get_thread_db(self) -> api.VectorDB:
        """Return self.db.

        All workers share the connection opened by task()'s `with self.db.init()`.
        Thread-safe DBs share it across multiple workers. Non-thread-safe DBs are
        clamped to max_workers=1, so there is never concurrent access.
        """
        return self.db

    def _insert_batch_with_retry(
        self,
        db: api.VectorDB,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        retry_idx: int = 0,
    ) -> int:
        """Insert a single batch with retry logic. Returns inserted count."""
        insert_kwargs = {
            "embeddings": embeddings,
            "metadata": metadata,
            "labels_data": labels_data,
        }
        if tenant_labels_data is not None:
            insert_kwargs["tenant_labels_data"] = tenant_labels_data
        insert_count, error = db.insert_embeddings(**insert_kwargs)
        if error is not None:
            log.warning(f"Insert failed, try_idx={retry_idx}, Exception: {error}")
            if getattr(error, "non_retryable", False):
                msg = f"Non-retryable insert failure after {insert_count} inserted rows: {error}"
                raise RuntimeError(msg) from error
            retry_idx += 1
            if retry_idx <= config.MAX_INSERT_RETRY:
                time.sleep(retry_idx)
                return self._insert_batch_with_retry(
                    db,
                    embeddings,
                    metadata,
                    labels_data,
                    tenant_labels_data,
                    retry_idx,
                )
            msg = f"Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
            raise RuntimeError(msg)
        return insert_count

    def _worker_insert(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
    ) -> int:
        """Worker function: insert a batch with retry."""
        db = self._get_thread_db()
        return self._insert_batch_with_retry(db, embeddings, metadata, labels_data, tenant_labels_data)

    def _next_batch(self) -> tuple[list[list[float]], list[int], list[str] | None, list[str] | None] | None:
        """Pull the next batch from the shared dataset iterator.

        Thread-safe: only one thread reads from the iterator at a time.
        Returns None when the iterator is exhausted.
        """
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            return None
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            return None
        with self._iter_lock:
            stop_event = getattr(self, "_stop_event", None)
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                data_df = next(self._dataset_iter)
            except StopIteration:
                return None

        all_metadata = data_df[self.dataset.data.train_id_field].tolist()
        emb_np = np.stack(data_df[self.dataset.data.train_vector_field])
        if self.normalize:
            all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
        else:
            all_embeddings = emb_np.tolist()
        del emb_np

        labels_data = None
        if self.filters.type == FilterOp.StrEqual or self.with_scalar_labels:
            label_field = self.filters.label_field if self.filters.type == FilterOp.StrEqual else "labels"
            if self.dataset.data.scalar_labels_file_separated:
                labels_data = self.dataset.scalar_labels[label_field][all_metadata].to_list()
            else:
                labels_data = data_df[label_field].tolist()

        tenant_labels_data = None
        if self.tenant_case is not None and getattr(self.tenant_case, "is_multitenant", False):
            tenant_labels_data = self.tenant_case.tenant_labels_for_ids(all_metadata)

        return all_embeddings, all_metadata, labels_data, tenant_labels_data

    def _worker_loop(self) -> int:
        """Worker loop: pull batches from the shared iterator and insert them."""
        total = 0
        try:
            while True:
                batch = self._next_batch()
                if batch is None:
                    break
                embeddings, metadata, labels_data, tenant_labels_data = batch
                total += self._worker_insert(embeddings, metadata, labels_data, tenant_labels_data)
        except Exception:
            stop_event = getattr(self, "_stop_event", None)
            if stop_event is not None:
                stop_event.set()
            raise
        return total

    def task(self) -> int:
        """Insert entire dataset using concurrent executor. Runs in subprocess."""
        count = 0
        self._iter_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._deadline = None if self.duration is None else time.perf_counter() + self.duration
        self._dataset_iter = self.dataset.iter_batches(self.batch_size)

        with self.db.init():
            log.info(
                f"({mp.current_process().name:16}) Start concurrent insert, "
                f"batch_size={self.batch_size}, max_workers={self.max_workers}"
            )
            start = time.perf_counter()

            with self._create_executor() as executor:
                for _ in range(self.max_workers):
                    executor.submit(self._worker_loop)

                batch_results = executor.wait_all()

            # Log all errors, then raise the first one
            errors = [r.error for r in batch_results if r.error is not None]
            if errors:
                for err in errors:
                    log.warning(f"Batch insert error: {err}")
                raise errors[0]

            count = sum(r.value for r in batch_results)

            log.info(
                f"({mp.current_process().name:16}) Finish concurrent insert, "
                f"count={count}, dur={time.perf_counter() - start:.2f}s"
            )
        return count

    @time_it
    def _insert_all_batches(self) -> int:
        """Performance case only: run task() in subprocess with timeout."""
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
                kill_proc_tree(pids=list(executor._processes.keys()))
                raise PerformanceTimeoutError(msg) from e
            except Exception as e:
                log.warning(f"VectorDB load dataset error: {e}")
                raise e from e
            else:
                return count

    def run(self) -> int:
        """Insert full dataset concurrently. Returns total inserted count."""
        count, _ = self._insert_all_batches()
        return count


class ConcurrentFtsInsertRunner:
    """Concurrent FTS document insert runner.

    This mirrors ConcurrentInsertRunner's threading model, but streams FTS
    documents and calls VectorDB.insert_documents() instead of insert_embeddings().
    The task runs in the current process because ir_datasets objects are not
    reliably picklable under spawn.
    """

    def __init__(
        self,
        db: api.VectorDB,
        dataset: FtsDatasetManager,
        timeout: float | None = None,
        max_workers: int | None = None,
        backend: ExecutorBackend = ExecutorBackend.THREADING,
        batch_size: int = config.NUM_PER_BATCH,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.backend = backend
        self.batch_size = batch_size

        effective_workers = max_workers or min(mp.cpu_count(), 4)
        if not db.thread_safe:
            log.info(f"DB {db.name} is not thread-safe, falling back to max_workers=1")
            effective_workers = 1
        self.max_workers = effective_workers
        assert db.thread_safe or self.max_workers == 1, "Non-thread-safe DBs must use max_workers=1"

    def _create_executor(self) -> TaskExecutor:
        if self.backend == ExecutorBackend.ASYNC:
            return AsyncExecutor(max_workers=self.max_workers)
        return ThreadExecutor(max_workers=self.max_workers)

    def _get_thread_db(self) -> api.VectorDB:
        return self.db

    def _insert_batch_with_retry(
        self,
        db: api.VectorDB,
        texts: list[str],
        doc_ids: list[str],
        retry_idx: int = 0,
    ) -> int:
        insert_count, error = db.insert_documents(texts=texts, doc_ids=doc_ids)
        if error is not None:
            log.warning(f"FTS insert failed, try_idx={retry_idx}, Exception: {error}")
            if getattr(error, "non_retryable", False):
                msg = f"Non-retryable FTS insert failure after {insert_count} inserted rows: {error}"
                raise RuntimeError(msg) from error
            retry_idx += 1
            if retry_idx <= config.MAX_INSERT_RETRY:
                time.sleep(retry_idx)
                return self._insert_batch_with_retry(db, texts, doc_ids, retry_idx)
            msg = f"FTS insert failed and retried more than {config.MAX_INSERT_RETRY} times"
            raise RuntimeError(msg)
        return insert_count

    def _worker_insert(self, texts: list[str], doc_ids: list[str]) -> int:
        db = self._get_thread_db()
        return self._insert_batch_with_retry(db, texts, doc_ids)

    def _record_insert_progress(self, insert_count: int) -> None:
        with self._count_lock:
            self._inserted_count += insert_count
            while self._inserted_count >= self._next_log_count:
                log.info(f"Loaded {self._next_log_count} FTS documents into VectorDB")
                self._next_log_count += 100_000

    def _next_batch(self) -> tuple[list[str], list[str]] | None:
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            return None

        with self._iter_lock:
            stop_event = getattr(self, "_stop_event", None)
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                batch = next(self._dataset_iter)
            except StopIteration:
                return None

        doc_ids: list[str] = []
        texts: list[str] = []
        for doc in batch:
            doc_ids.append(str(doc.doc_id if hasattr(doc, "doc_id") else doc["doc_id"]))
            texts.append(doc.text if hasattr(doc, "text") else doc["text"])
        return texts, doc_ids

    def _worker_loop(self) -> int:
        total = 0
        try:
            while True:
                batch = self._next_batch()
                if batch is None:
                    break
                texts, doc_ids = batch
                inserted = self._worker_insert(texts, doc_ids)
                total += inserted
                self._record_insert_progress(inserted)
        except Exception:
            stop_event = getattr(self, "_stop_event", None)
            if stop_event is not None:
                stop_event.set()
            raise
        return total

    def task(self) -> int:
        count = 0
        self._iter_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._inserted_count = 0
        self._next_log_count = 100_000
        self._dataset_iter = self.dataset.iter_batches(self.batch_size)

        with self.db.init():
            log.info(
                f"Start concurrent FTS insert, batch_size={self.batch_size}, max_workers={self.max_workers}"
            )
            start = time.perf_counter()

            with self._create_executor() as executor:
                for _ in range(self.max_workers):
                    executor.submit(self._worker_loop)

                batch_results = executor.wait_all()

            errors = [r.error for r in batch_results if r.error is not None]
            if errors:
                for err in errors:
                    log.warning(f"FTS batch insert error: {err}")
                raise errors[0]

            count = sum(r.value for r in batch_results)

            log.info(
                f"Finish concurrent FTS insert, count={count}, dur={time.perf_counter() - start:.2f}s"
            )
        return count

    @time_it
    def _insert_all_batches(self) -> int:
        return self.task()

    def run(self) -> int:
        with run_timeout(self.timeout, lambda: LoadTimeoutError(self.timeout)):
            count, _ = self._insert_all_batches()
        return count
