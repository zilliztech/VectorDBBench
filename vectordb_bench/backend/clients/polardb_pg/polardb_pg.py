"""PolarDB for PostgreSQL HNSW client."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from psycopg import sql

from ..pgvector.pgvector import PgVector

if TYPE_CHECKING:
    from .config import PolarDBPgHNSWConfig

log = logging.getLogger(__name__)

_GRAPH_CACHE_POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class HNSWGraphCacheDetail:
    requested: str
    status: str
    usable: bool
    last_error: str | None
    raw: str

    @staticmethod
    def _field(detail: str, name: str) -> str:
        match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", detail)
        if match is None:
            msg = f"Graph Cache detail is missing {name}: {detail}"
            raise RuntimeError(msg)
        return match.group(1)

    @classmethod
    def parse(cls, detail: str) -> HNSWGraphCacheDetail:
        requested = cls._field(detail, "requested")
        status = cls._field(detail, "status")
        usable_value = cls._field(detail, "usable")
        if usable_value not in {"yes", "no"}:
            msg = f"Graph Cache detail has invalid usable state: {detail}"
            raise RuntimeError(msg)
        error_match = re.search(r"(?:^|\s)last_error=(.*)$", detail)
        last_error = error_match.group(1) if error_match else None
        if last_error == "none":
            last_error = None
        return cls(
            requested=requested,
            status=status,
            usable=usable_value == "yes",
            last_error=last_error,
            raw=detail,
        )


class PolarDBPgHNSW(PgVector):
    """PgVector-compatible PolarDB HNSW client."""

    name = "PolarDBPG"
    case_config: PolarDBPgHNSWConfig

    def __init__(self, *args, drop_old: bool = False, **kwargs):
        super().__init__(*args, drop_old=drop_old, **kwargs)

        # Search-only benchmarks skip optimize(), so validate and prepare the
        # existing index before any search workers can start.
        if not drop_old and self.case_config.graph_cache:
            with self.init():
                self._ensure_graph_cache_ready()

    def _post_insert(self):
        super()._post_insert()
        if self.case_config.graph_cache:
            self._ensure_graph_cache_ready()

    def _qualified_index_name(self) -> str:
        return f"public.{self._index_name}"

    def _graph_cache_detail(self) -> HNSWGraphCacheDetail:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        row = self.cursor.execute(
            "SELECT hnsw_cache_detail(%s::regclass)",
            (self._qualified_index_name(),),
        ).fetchone()
        self.conn.commit()
        if row is None or row[0] is None:
            msg = f"Unable to read Graph Cache state for {self._qualified_index_name()}"
            raise RuntimeError(msg)
        return HNSWGraphCacheDetail.parse(row[0])

    def _call_graph_cache_function(self, function_name: str) -> bool:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        statement = sql.SQL("SELECT {}(%s::regclass)").format(sql.Identifier(function_name))
        row = self.cursor.execute(statement, (self._qualified_index_name(),)).fetchone()
        self.conn.commit()
        return bool(row and row[0])

    def _check_graph_cache_role(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        row = self.cursor.execute("SELECT pg_is_in_recovery()").fetchone()
        self.conn.commit()
        if row and row[0]:
            msg = "PolarDB HNSW Graph Cache benchmarks must run on the primary node"
            raise RuntimeError(msg)

    def _ensure_graph_cache_ready(self):
        self._check_graph_cache_role()
        timeout = self.case_config.graph_cache_timeout
        deadline = time.monotonic() + timeout
        last_logged_state: tuple[str, bool] | None = None
        last_progress_log = 0.0
        release_requested = False

        while True:
            detail = self._graph_cache_detail()
            if detail.requested != "on":
                msg = (
                    f"HNSW index {self._qualified_index_name()} was not created with cache=on; "
                    "enable that reloption, rebuild the index, or run with --skip-graph-cache"
                )
                raise RuntimeError(msg)
            if detail.status == "not_preloaded":
                msg = (
                    "HNSW Graph Cache is not preloaded; add vector to "
                    "shared_preload_libraries and restart PostgreSQL"
                )
                raise RuntimeError(msg)
            if detail.status == "ready" and detail.usable:
                log.info("HNSW Graph Cache is ready for %s", self._qualified_index_name())
                return
            now = time.monotonic()
            state = (detail.status, detail.usable)
            if state != last_logged_state or now - last_progress_log >= 60:
                log.info(
                    "Waiting for HNSW Graph Cache: index=%s status=%s usable=%s",
                    self._qualified_index_name(),
                    detail.status,
                    "yes" if detail.usable else "no",
                )
                last_logged_state = state
                last_progress_log = now

            if detail.status == "ready":
                if not release_requested:
                    log.info("Rebuilding stale HNSW Graph Cache for %s", self._qualified_index_name())
                    self._call_graph_cache_function("hnsw_release_cache")
                    release_requested = True
            elif detail.status == "empty":
                if detail.last_error:
                    msg = f"HNSW Graph Cache build failed for {self._qualified_index_name()}: {detail.last_error}"
                    raise RuntimeError(msg)
                accepted = self._call_graph_cache_function("hnsw_schedule_cache_rebuild")
                if accepted:
                    release_requested = False
                    log.info("Scheduled HNSW Graph Cache build for %s", self._qualified_index_name())
            elif detail.status not in {"building", "draining"}:
                msg = f"Unsupported HNSW Graph Cache state for {self._qualified_index_name()}: {detail.raw}"
                raise RuntimeError(msg)

            if now >= deadline:
                msg = (
                    f"Timed out after {timeout:g}s waiting for "
                    f"HNSW Graph Cache on {self._qualified_index_name()}: {detail.raw}"
                )
                raise TimeoutError(msg)
            time.sleep(min(_GRAPH_CACHE_POLL_INTERVAL_SECONDS, max(deadline - now, 0)))
