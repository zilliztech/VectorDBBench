import contextlib
import logging
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import psutil

log = logging.getLogger(__name__)


def kill_proc_tree(pids: list[int] | None = None, grace: float = 2, timeout: float = 3):
    """Kill child processes with SIGTERM, then SIGKILL for survivors.

    Args:
        pids: Specific PIDs to kill. If None, kills all children of the
              current process (recursive).
        grace: Seconds to wait after SIGTERM before sending SIGKILL.
        timeout: Seconds to wait for processes to fully exit after SIGKILL.
    """
    if pids is not None:
        targets = []
        for pid in pids:
            with contextlib.suppress(psutil.NoSuchProcess):
                targets.append(psutil.Process(pid))
    else:
        targets = psutil.Process().children(recursive=True)

    for p in targets:
        try:
            log.warning(f"sending SIGTERM to child process: {p}")
            p.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(targets, timeout=grace)
    for p in alive:
        try:
            log.warning(f"force killing child process: {p}")
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=timeout)


def numerize(n: int) -> str:
    """display positive number n for readability

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
        "END": float("inf"),
    }

    display_n, sufix = n, ""
    for s, base in sufix2upbound.items():
        # number >= 1000B will alway have sufix 'B'
        if s == "END":
            display_n = int(n / 1e9)
            sufix = "B"
            break

        if n < base:
            sufix = "" if s == "EMPTY" else s
            display_n = int(n / (base / 1e3))
            break
    return f"{display_n}{sufix}"


def time_it(func: any):
    """returns result and elapsed time"""

    @wraps(func)
    def inner(*args, **kwargs):
        pref = time.perf_counter()
        result = func(*args, **kwargs)
        delta = time.perf_counter() - pref
        return result, delta

    return inner


@contextmanager
def timeout(timeout_seconds: float | None, exc_factory: Callable[[], Exception]):
    if (
        timeout_seconds is None
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _handle_timeout(signum, frame):  # noqa: ANN001, ARG001
        raise exc_factory()

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def compose_train_files(train_count: int, use_shuffled: bool) -> list[str]:
    prefix = "shuffle_train" if use_shuffled else "train"
    middle = f"of-{train_count}"
    surfix = "parquet"

    train_files = []
    if train_count > 1:
        just_size = 2
        for i in range(train_count):
            sub_file = f"{prefix}-{str(i).rjust(just_size, '0')}-{middle}.{surfix}"
            train_files.append(sub_file)
    else:
        train_files.append(f"{prefix}.{surfix}")

    return train_files


ONE_PERCENT = 0.01
NINETY_NINE_PERCENT = 0.99


def compose_gt_file(filters: float | str | None = None) -> str:
    if filters is None:
        return "neighbors.parquet"

    if filters == ONE_PERCENT:
        return "neighbors_head_1p.parquet"

    if filters == NINETY_NINE_PERCENT:
        return "neighbors_tail_1p.parquet"

    msg = f"Filters not supported: {filters}"
    raise ValueError(msg)


MASK = "**********"

SENSITIVE_KEYS = frozenset({"api_key", "password", "token"})


def redact_sensitive(value: Any) -> Any:
    """Recursively replace credentials in a config structure with a fixed mask.

    ``DBConfig.to_dict()`` hands the driver its plaintext credentials, so its result
    must be redacted before it reaches a log record or a result file.

    Examples:
        >>> redact_sensitive({"host": "h", "password": "s3cret"})
        {'host': 'h', 'password': '**********'}
        >>> redact_sensitive({"http_auth": ("admin", "s3cret")})
        {'http_auth': ('admin', '**********')}
    """
    if isinstance(value, dict):
        return {key: _redact_entry(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def _redact_entry(key: Any, value: Any) -> Any:
    name = key.lower() if isinstance(key, str) else key
    if name in SENSITIVE_KEYS and value:
        return MASK
    if name == "http_auth":
        return _redact_http_auth(value)
    return redact_sensitive(value)


def _redact_http_auth(value: Any) -> Any:
    """http_auth is either a (user, secret) pair or an opaque auth object (e.g. AWS4Auth)."""
    if isinstance(value, tuple | list) and len(value) == 2:
        redacted = (value[0], MASK)
        return list(redacted) if isinstance(value, list) else redacted
    return MASK if value else value
