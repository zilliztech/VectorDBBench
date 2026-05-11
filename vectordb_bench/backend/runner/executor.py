"""Task executor abstraction with threading and async backends.

Provides a unified interface for submitting callables with controlled
concurrency. Two implementations:
  - ThreadExecutor: backed by ThreadPoolExecutor
  - AsyncExecutor: backed by asyncio with semaphore-based concurrency control
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single submitted task."""

    value: Any = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class TaskExecutor(ABC):
    """Abstract executor that accepts callables and controls concurrency."""

    @abstractmethod
    def start(self) -> None:
        """Initialize executor resources."""
        raise NotImplementedError

    @abstractmethod
    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Submit a task for execution."""
        raise NotImplementedError

    @abstractmethod
    def wait_all(self) -> list[TaskResult]:
        """Block until all submitted tasks complete. Return results in submission order."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release executor resources. Safe to call multiple times."""
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> bool:
        self.shutdown()
        return False


class ThreadExecutor(TaskExecutor):
    """ThreadPoolExecutor-backed implementation."""

    def __init__(self, max_workers: int):
        self._max_workers = max(1, max_workers)
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future] = []

    def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._futures = []

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._executor is None:
            raise RuntimeError("Executor not started. Call start() or use as context manager.")
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)

    def wait_all(self) -> list[TaskResult]:
        results = []
        for future in self._futures:
            try:
                value = future.result()
                results.append(TaskResult(value=value))
            except Exception as e:
                results.append(TaskResult(error=e))
        self._futures = []
        return results

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class AsyncExecutor(TaskExecutor):
    """asyncio-backed implementation for async DB clients.

    Accepts coroutine functions (async def), runs them on a single event
    loop thread with semaphore-based concurrency control. No thread pool.
    """

    def __init__(self, max_workers: int):
        self._max_workers = max(1, max_workers)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._coros: list = []
        self._owns_loop = False

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        self._semaphore = asyncio.Semaphore(self._max_workers)
        self._coros = []

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Submit a callable for execution.

        Accepts both coroutine functions (async def) and regular functions.
        Sync functions are offloaded to a thread via run_in_executor.
        """
        if self._loop is None or self._semaphore is None:
            raise RuntimeError("Executor not started. Call start() or use as context manager.")

        async def _run():
            async with self._semaphore:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        self._coros.append(_run())

    def wait_all(self) -> list[TaskResult]:
        if not self._coros:
            return []

        async def _gather():
            gathered = await asyncio.gather(*self._coros, return_exceptions=True)
            results = []
            for item in gathered:
                if isinstance(item, Exception):
                    results.append(TaskResult(error=item))
                else:
                    results.append(TaskResult(value=item))
            return results

        if self._owns_loop:
            results = self._loop.run_until_complete(_gather())
        else:
            results = asyncio.run_coroutine_threadsafe(_gather(), self._loop).result()

        self._coros = []
        return results

    def shutdown(self) -> None:
        if self._owns_loop and self._loop is not None:
            self._loop.close()
        self._loop = None
        self._semaphore = None
