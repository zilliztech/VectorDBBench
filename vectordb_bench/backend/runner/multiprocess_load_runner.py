"""Multi-process loader for the performance load stage."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ... import config
from ...models import PerformanceTimeoutError
from ..filter import Filter, FilterOp, non_filter
from ..utils import time_it

if TYPE_CHECKING:
    from multiprocessing.queues import Queue as MPQueue
    from multiprocessing.sharedctypes import Synchronized

    from ..clients import api
    from ..dataset import DatasetManager


log = logging.getLogger(__name__)


# Sent from the producer to signal "no more work".
_SENTINEL = None

# Poll interval while the producer waits for a stalled queue slot.
_PROGRESS_INTERVAL_SEC = 5


def _insert_worker(
    db: api.VectorDB,
    task_queue: MPQueue,
    error_queue: MPQueue,
    counter: Synchronized,
    normalize: bool,
    worker_id: int,
) -> None:
    """Drain `task_queue` in a dedicated process.

    Each item is a `(ids, emb_np, labels)` tuple. Workers share no state
    beyond the queues + counter; a failure from any worker is surfaced via
    `error_queue` and causes the runner to abort.
    """
    try:
        with db.init():
            while True:
                item = task_queue.get()
                if item is _SENTINEL:
                    return

                ids, emb_np, labels = item
                if normalize:
                    emb_np = emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]
                embeddings = emb_np.tolist()

                kwargs: dict[str, Any] = {}
                if labels is not None:
                    kwargs["labels_data"] = labels
                inserted, err = db.insert_embeddings(
                    embeddings=embeddings,
                    metadata=ids,
                    **kwargs,
                )
                if err is not None:
                    raise err  # noqa: TRY301
                with counter.get_lock():
                    counter.value += inserted
    except Exception as exc:
        log.exception(f"[worker-{worker_id}] insert failed")
        with contextlib.suppress(Exception):
            error_queue.put(f"worker-{worker_id}: {exc!r}")


class MultiprocessInsertRunner:
    """Run insert_embeddings across a pool of worker processes."""

    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        workers: int,
        filters: Filter = non_filter,
        timeout: float | None = None,
        queue_size: int | None = None,
    ):
        self.db = db
        self.dataset = dataset
        self.normalize = normalize
        self.filters = filters
        self.timeout = timeout if isinstance(timeout, int | float) else None

        if workers <= 0:
            workers = mp.cpu_count()
        self.workers = workers
        # A couple slots of slack per worker lets the producer run a step ahead
        # without unbounded memory growth.
        self.queue_size = queue_size if queue_size and queue_size > 0 else max(workers * 2, 4)

    def _pull_labels(self, data_df: Any, ids: list[int]) -> list | None:
        """Extract per-row scalar labels for StrEqual filter, if enabled."""
        if self.filters.type != FilterOp.StrEqual:
            return None
        if self.dataset.data.scalar_labels_file_separated:
            return self.dataset.scalar_labels[self.filters.label_field][ids].to_list()
        return data_df[self.filters.label_field].tolist()

    @time_it
    def _run(self) -> int:
        ctx = mp.get_context("spawn")
        task_queue: MPQueue = ctx.Queue(maxsize=self.queue_size)
        error_queue: MPQueue = ctx.Queue()
        counter: Synchronized = ctx.Value("q", 0)

        log.info(
            f"Multiprocess load start: workers={self.workers}, "
            f"queue_size={self.queue_size}, batch_size={config.NUM_PER_BATCH}",
        )

        procs: list[mp.Process] = []
        for i in range(self.workers):
            p = ctx.Process(
                target=_insert_worker,
                args=(
                    self.db,
                    task_queue,
                    error_queue,
                    counter,
                    self.normalize,
                    i,
                ),
                name=f"vdb-load-{i}",
                # daemon so workers die with the parent subprocess on SIGTERM/SIGKILL.
                daemon=True,
            )
            p.start()
            procs.append(p)

        id_field = self.dataset.data.train_id_field
        vec_field = self.dataset.data.train_vector_field

        produced = 0
        start = time.perf_counter()
        last_log = start
        interrupted = False

        try:
            for data_df in self.dataset:
                if self._abort_if_worker_died(error_queue, procs):
                    break

                ids = data_df[id_field].tolist()
                emb_np = np.stack(data_df[vec_field])
                labels = self._pull_labels(data_df, ids)
                # Short timeout on put so a stuck queue doesn't block SIGTERM/Ctrl+C.
                self._put_with_interruptible_wait(task_queue, (ids, emb_np, labels), procs)
                produced += len(ids)

                now = time.perf_counter()
                if now - last_log >= _PROGRESS_INTERVAL_SEC:
                    done = counter.value
                    elapsed = now - start
                    rate = done / elapsed if elapsed > 0 else 0
                    log.info(
                        f"Load progress: produced={produced} inserted={done} "
                        f"rate={rate:.0f}/s elapsed={elapsed:.0f}s",
                    )
                    last_log = now

                if self.timeout is not None and (now - start) > self.timeout:
                    msg = f"Multiprocess load exceeded timeout {self.timeout}s"
                    log.warning(msg)
                    raise PerformanceTimeoutError(msg)
        except KeyboardInterrupt:
            log.warning("Multiprocess load interrupted by user")
            interrupted = True
            raise
        finally:
            self._shutdown(procs, task_queue, graceful=not interrupted)

        if not error_queue.empty():
            err_detail = error_queue.get_nowait()
            msg = f"Multiprocess load failed: {err_detail}"
            raise RuntimeError(msg)

        inserted = counter.value
        elapsed = time.perf_counter() - start
        rate = inserted / elapsed if elapsed > 0 else 0
        log.info(
            f"Multiprocess load done: produced={produced} inserted={inserted} "
            f"elapsed={elapsed:.2f}s rate={rate:.0f}/s",
        )
        return inserted

    @staticmethod
    def _put_with_interruptible_wait(
        queue: MPQueue,
        item: Any,
        procs: list[mp.Process],
        chunk_timeout: float = 1.0,
    ) -> None:
        """`queue.put` that polls so a Ctrl+C or worker crash doesn't block it forever."""
        while True:
            try:
                queue.put(item, timeout=chunk_timeout)
            except Exception:
                if all(not p.is_alive() for p in procs):
                    msg = "All workers exited while producer was waiting to enqueue"
                    raise RuntimeError(msg) from None
            else:
                return

    @staticmethod
    def _abort_if_worker_died(error_queue: MPQueue, procs: list[mp.Process]) -> bool:
        """Surface worker failures or unexpected exits early."""
        if not error_queue.empty():
            return True
        return any(not p.is_alive() and p.exitcode not in (None, 0) for p in procs)

    @staticmethod
    def _shutdown(
        procs: list[mp.Process],
        task_queue: MPQueue,
        graceful: bool,
        graceful_join: float = 10.0,
        hard_join: float = 3.0,
    ) -> None:
        """Wind down workers. On interrupt, skip draining and terminate fast."""
        if graceful:
            # Let workers finish what's already queued, then exit on sentinel.
            for _ in procs:
                try:
                    task_queue.put(_SENTINEL, timeout=5)
                except Exception:
                    break
            for p in procs:
                p.join(timeout=graceful_join)

        # Anyone still alive gets terminated — includes the interrupted path.
        for p in procs:
            if p.is_alive():
                if graceful:
                    log.warning(f"worker {p.name} did not exit cleanly; terminating")
                with contextlib.suppress(Exception):
                    p.terminate()
        for p in procs:
            p.join(timeout=hard_join)
            if p.is_alive():
                with contextlib.suppress(Exception):
                    p.kill()
                p.join(timeout=1)

    def run(self) -> int:
        count, _ = self._run()
        return count
