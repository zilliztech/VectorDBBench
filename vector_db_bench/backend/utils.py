import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable


def numerize(n) -> str:
    """display positive number in for readability

    Examples:
        >>> numerize(1_000)
        '1K'
        >>> numerize(1_000_000_000)
        '1B'
    """
    sufix2upbound = {
        "EMPTY": 1e3,
        "K": 1e6,
        "M": 1e9,
        "B": 1e12,
        "END": float('inf'),
    }

    display_n, sufix = n, ""
    for s, base in sufix2upbound.items():
        # number >= 1000B will alway has sufix 'B'
        if s == "END":
            display_n = int(n/1e9)
            sufix = "B"
            break

        if n < base:
            sufix = "" if s == "EMPTY" else s
            display_n = int(n/(base/1e3))
            break
    return f"{display_n}{sufix}"


@dataclass
class Timer(ContextDecorator):
    """simple timer """

    text: str = "elapsed time: {:0.4f}s"
    name: str | None = None
    logger: Callable[[str], None] = print
    _start_time: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(f"{self.name}: {self.text.format(elapsed_time)}")

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
