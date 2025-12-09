"""
Database retry utilities for handling CockroachDB transient connection failures.

Implements retry logic for CockroachDB-specific error codes:
- 40001: Serialization failure
- 40003: Statement completion unknown (multi-region ambiguous results)
"""

import functools
import logging
import time
from collections.abc import Callable

from psycopg import InterfaceError, OperationalError

log = logging.getLogger(__name__)

TRANSIENT_ERRORS = (
    OperationalError,
    InterfaceError,
    Exception,  # Catch all for transient detection
)

TRANSIENT_ERROR_MESSAGES = (
    "server closed the connection unexpectedly",
    "connection already closed",
    "SSL connection has been closed unexpectedly",
    "could not receive data from server",
    "connection timed out",
    "Connection refused",
    "connection reset by peer",
    "broken pipe",
    "restart transaction",
    "TransactionRetryError",
    "SerializationFailure",
    "StatementCompletionUnknown",
    "result is ambiguous",
    "failed to connect",
    "no such host",
    "initial connection heartbeat failed",
    "sending to all replicas failed",
    "transaction is aborted",
    "commands ignored until end of transaction block",
    "40001",
    "40003",
)


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried."""
    if not isinstance(error, TRANSIENT_ERRORS):
        return False

    error_msg = str(error).lower()
    return any(msg.lower() in error_msg for msg in TRANSIENT_ERROR_MESSAGES)


def db_retry(
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    exceptions: tuple[type[Exception], ...] = TRANSIENT_ERRORS,
):
    """
    Decorator to retry database operations on transient failures.

    Uses exponential backoff with configurable parameters per CockroachDB best practices.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_delay: Initial retry delay in seconds (default: 0.5)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries (default: 10.0)
        exceptions: Tuple of exception types to catch and retry
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    attempt += 1

                    if not is_transient_error(e):
                        log.warning(f"Non-transient error in {func.__name__}: {e}")
                        raise

                    if attempt >= max_attempts:
                        log.warning(f"Max retry attempts ({max_attempts}) reached for {func.__name__}")
                        raise

                    current_delay = min(delay * (backoff_factor ** (attempt - 1)), max_delay)

                    log.info(
                        f"Transient DB error in {func.__name__} (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )

                    time.sleep(current_delay)

            msg = f"Unexpected state in retry logic for {func.__name__}"
            raise RuntimeError(msg)

        return wrapper

    return decorator
