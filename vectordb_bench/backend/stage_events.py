"""Opt-in, line-oriented benchmark phase events for external collectors."""

import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

PREFIX = "E2E_STAGE "
_parent = ContextVar("e2e_stage_parent", default=None)


def enabled():
    return os.environ.get("E2E_STAGE_EVENTS") == "1"


def emit(stage: str, state: str, span_id: str | None, **fields):
    if not enabled():
        return
    event = {
        "schema": "milvus-e2e-stage/v1",
        "run_id": os.environ.get("E2E_RUN_ID", ""),
        "case_id": os.environ.get("E2E_CASE_ID", ""),
        "attempt": int(os.environ.get("E2E_ATTEMPT", "1")),
        "stage": stage,
        "state": state,
        "span_id": span_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    # One bounded write prevents concurrent worker events from being interleaved.
    data = (PREFIX + json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 4096:
        raise ValueError("Stage event exceeds atomic pipe write size")
    os.write(1, data)


@contextmanager
def stage(name: str, **fields):
    if not enabled():
        yield None
        return
    span_id = uuid.uuid4().hex
    if _parent.get():
        fields["parent_span_id"] = _parent.get()
    emit(name, "start", span_id, **fields)
    token = _parent.set(span_id)
    try:
        yield span_id
    except BaseException:
        emit(name, "failed", span_id, **fields)
        raise
    else:
        emit(name, "end", span_id, **fields)
    finally:
        _parent.reset(token)
