"""HouyiDB client for VectorDBBench.

HouyiDB exposes two data planes:

* pgwire (PostgreSQL protocol) with a compact BYTEA batch format for vector
  upserts and queries;
* an HTTP benchmark API for bulk upserts, filtered search, health checks,
  optimize and index status.

This client uses pgwire for unfiltered workloads and the HTTP API when scalar
labels or id filters are required. Both paths target the same server-side
index. A fresh server (empty WAL) is required for a cold run because the
server has no client-side "drop collection" operation for the main index:
``drop_old=True`` is enforced by rejecting an occupied index instead of
silently loading on top of stale data.

Search failures are propagated to VectorDBBench's runner so a failed request
is never counted as a successful (and extremely fast) query. HouyiDB builds a
cosine index, so datasets using another distance metric are rejected instead
of being benchmarked against incompatible ground truth.
"""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import requests

from vectordb_bench.backend.filter import Filter, FilterOp, non_filter
from vectordb_bench.backend.payload import PayloadProfile

from ..api import MetricType, VectorDB

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from .config import HouyiDBIndexConfig

log = logging.getLogger(__name__)


def _encode_upsert_batch(dim: int, items: list[tuple[int, list[float]]]) -> bytes:
    """Encode the AUB1 BYTEA payload understood by the pgwire upsert directive."""
    out = bytearray(b"AUB1")
    out.extend(int(dim).to_bytes(4, "little"))
    out.extend(len(items).to_bytes(4, "little"))
    for vector_id, vector in items:
        if len(vector) != dim:
            msg = f"AUB1 dim mismatch: expected {dim}, got {len(vector)}"
            raise ValueError(msg)
        out.extend(int(vector_id).to_bytes(8, "little"))
        for value in vector:
            out.extend(struct.pack("<f", float(value)))
    return bytes(out)


def _encode_query_vector(values: list[float]) -> bytes:
    """Encode the AVB1 BYTEA payload understood by the pgwire query directive."""
    out = bytearray(b"AVB1")
    out.extend(len(values).to_bytes(4, "little"))
    for value in values:
        out.extend(struct.pack("<f", float(value)))
    return bytes(out)


class _InflightPgwireQuery:
    """One pgwire statement under a hard client-side deadline."""

    __slots__ = ("cancel_called", "close_called", "close_deadline", "conn", "deadline", "timed_out")

    def __init__(self, conn: Any, deadline: float, close_deadline: float) -> None:
        self.conn = conn
        self.deadline = deadline
        self.close_deadline = close_deadline
        self.timed_out = threading.Event()
        self.cancel_called = False
        self.close_called = False

    def abort(self, now: float) -> None:
        if now >= self.deadline and not self.cancel_called:
            self.cancel_called = True
            self.timed_out.set()
            self._cancel()
        # A cancellation that does not unblock libpq (dead peer, half-open
        # socket) is followed by closing the connection: the only remaining
        # way to bound the call.
        if now >= self.close_deadline and not self.close_called:
            self.close_called = True
            with contextlib.suppress(Exception):
                # Best-effort unblock; close is the last resort.
                self.conn.close()

    def _cancel(self) -> None:
        cancel_safe = getattr(self.conn, "cancel_safe", None)
        if callable(cancel_safe):
            try:
                cancel_safe(timeout=5)
            except TypeError:
                try:
                    cancel_safe()
                except Exception:
                    log.debug("HouyiDB pgwire cancel_safe() failed", exc_info=True)
                else:
                    return
            except Exception:
                log.debug("HouyiDB pgwire cancel_safe(timeout=5) failed", exc_info=True)
            else:
                return
        cancel = getattr(self.conn, "cancel", None)
        if callable(cancel):
            with contextlib.suppress(Exception):
                # Legacy psycopg 3.1 fallback; abort() closes the connection
                # after the grace period if cancellation does not unblock.
                cancel()


class _PgwireWatchdog:
    """Cancel pgwire statements that outlive ``timeout_sec``.

    psycopg's ``connect_timeout`` only bounds connection establishment, so a
    blocked statement would otherwise never expire. One daemon thread per
    client instance scans the in-flight statements, which is much cheaper
    than creating a timer thread per query at benchmark rates.
    """

    _POLL_SECONDS = 0.05
    _CLOSE_GRACE_SECONDS = 5.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[int, _InflightPgwireQuery] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="houyidb-pgwire-watchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None

    def register(self, conn: Any, timeout_sec: float) -> _InflightPgwireQuery:
        now = time.monotonic()
        token = _InflightPgwireQuery(
            conn,
            now + timeout_sec,
            now + timeout_sec + self._CLOSE_GRACE_SECONDS,
        )
        with self._lock:
            self._inflight[id(token)] = token
        return token

    def unregister(self, token: _InflightPgwireQuery) -> None:
        with self._lock:
            self._inflight.pop(id(token), None)

    def _run(self) -> None:
        while not self._stop.wait(self._POLL_SECONDS):
            now = time.monotonic()
            with self._lock:
                tokens = list(self._inflight.values())
            for token in tokens:
                if now >= token.deadline:
                    token.abort(now)


class HouyiDB(VectorDB):
    """VectorDBBench client for HouyiDB."""

    thread_safe: bool = True
    supported_filter_types = [FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual]

    _HTTP_UPSERT_CHUNK_SIZE = 64
    _PGWIRE_USER = "tester"
    _PGWIRE_DATABASE = "houyi"

    def __init__(
        self,
        dim: int,
        db_config: dict[str, Any],
        db_case_config: HouyiDBIndexConfig,
        collection_name: str = "houyidb_collection",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name = "HouyiDB"
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.with_scalar_labels = with_scalar_labels
        self.transport = str(db_config.get("transport", "pgwire")).lower()
        if self.transport not in ("http", "pgwire"):
            msg = f"unsupported HouyiDB transport={self.transport!r}; expected 'pgwire' or 'http'"
            raise ValueError(msg)
        self.metric = self.case_config.metric_type or MetricType.COSINE
        if self.metric is not MetricType.COSINE:
            msg = (
                f"HouyiDB builds a cosine index; dataset metric {self.metric.name} is not supported. "
                "VectorDBBench would otherwise measure cosine search against "
                f"{self.metric.name} ground truth."
            )
            raise ValueError(msg)
        self.timeout_sec = int(db_config.get("timeout_sec", 600))
        self.verify_build_params = bool(db_config.get("verify_build_params", True))
        raw_api_key = db_config.get("api_key")
        self.api_key = raw_api_key.get_secret_value() if hasattr(raw_api_key, "get_secret_value") else raw_api_key
        scheme = "https" if db_config.get("use_tls", False) else "http"
        http_port = db_config.get("http_port")
        self._http_port = int(http_port) if http_port is not None else int(db_config["port"])
        host = str(db_config["host"])
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self.base_url = f"{scheme}://{host}:{self._http_port}"

        self._thread_local = threading.local()
        self._http_sessions_lock = threading.Lock()
        self._http_sessions: list[requests.Session] = []
        self._pgwire_sessions_lock = threading.Lock()
        self._pgwire_sessions: list[Any] = []
        self._pgwire_watchdog = _PgwireWatchdog()
        self._filter: Filter = non_filter
        self._connected = False
        self._build_params_verified = False
        self._build_params_warned = False

        # VectorDBBench constructs the client once per fresh load and pickles
        # it into search workers afterwards, so the emptiness check belongs
        # here: `init()` also runs inside those workers, where the index is
        # legitimately populated.
        if drop_old:
            self._ensure_empty_index()
        log.info(
            "HouyiDB initialized transport=%s dim=%s index=%s",
            self.transport,
            self.dim,
            self.case_config.index_param(),
        )
        log.info("HouyiDB search params=%s", self.case_config.search_param())

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for key in (
            "_thread_local",
            "_http_sessions_lock",
            "_http_sessions",
            "_pgwire_sessions_lock",
            "_pgwire_sessions",
            "_pgwire_watchdog",
        ):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._thread_local = threading.local()
        self._http_sessions_lock = threading.Lock()
        self._http_sessions = []
        self._pgwire_sessions_lock = threading.Lock()
        self._pgwire_sessions = []
        self._pgwire_watchdog = _PgwireWatchdog()
        # Each worker process must observe the server's effective parameters
        # itself; a parent-process success is not a proof for this process.
        self._build_params_verified = False
        self._build_params_warned = False

    def _http_session(self) -> requests.Session:
        if not hasattr(self._thread_local, "session"):
            session = requests.Session()
            if self.api_key:
                session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            with self._http_sessions_lock:
                self._http_sessions.append(session)
            try:
                response = session.get(f"{self.base_url}/v1/health", timeout=self.timeout_sec)
                response.raise_for_status()
            except Exception as exc:
                with self._http_sessions_lock:
                    if session in self._http_sessions:
                        self._http_sessions.remove(session)
                session.close()
                msg = f"failed to connect HouyiDB HTTP endpoint {self.base_url}: {exc}"
                raise RuntimeError(msg) from exc
            self._thread_local.session = session
        return self._thread_local.session

    def _index_status(self) -> dict[str, Any]:
        """Fetch the main-index status (hot/indexed/pending plus HNSW config)."""
        session = self._http_session()
        response = session.get(f"{self.base_url}/v1/index/status", timeout=self.timeout_sec)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            msg = f"unexpected HouyiDB index status payload: {payload!r}"
            raise TypeError(msg)
        return payload

    def _ensure_empty_index(self) -> None:
        """Enforce ``drop_old``: refuse a populated index instead of contaminating a run.

        HouyiDB has no remote drop operation for the main index. A fresh load
        on an occupied index would keep stale rows from an earlier case and
        silently mix two datasets, so the client fails loudly and tells the
        operator how to obtain a clean server.
        """
        status = self._index_status()
        occupied = sum(int(status.get(key, 0)) for key in ("hot_count", "indexed_count", "pending_count"))
        if occupied:
            msg = (
                "drop_old=True but the HouyiDB main index is not empty "
                f"(hot={status.get('hot_count')}, indexed={status.get('indexed_count')}, "
                f"pending={status.get('pending_count')}, collection={self.collection_name!r}). "
                "HouyiDB has no remote drop: stop the server, remove its WAL/data directory, "
                "restart it, and rerun the case."
            )
            raise RuntimeError(msg)

    def _verify_build_params(self) -> None:
        """Verify that the server's effective HNSW parameters match the case config.

        pgwire applies build parameters only from the server's startup flags
        (``--pgwire-hnsw-m`` / ``--pgwire-hnsw-efc``); the HTTP path applies the
        per-request hints. Either way, recording ``m``/``ef_construction`` from
        the case config without checking the server would mislabel the run, so
        this probes ``/v1/index/status`` after the load and before search.
        """
        if self._build_params_verified:
            return
        if not self.verify_build_params:
            if not self._build_params_warned:
                log.warning(
                    "HouyiDB build-parameter verification is disabled; recorded m=%s "
                    "ef_construction=%s may differ from the server's effective graph",
                    self.case_config.m,
                    self.case_config.ef_construction,
                )
                self._build_params_warned = True
            self._build_params_verified = True
            return

        status = self._index_status()
        missing = [key for key in ("hnsw_m", "hnsw_ef_construction") if key not in status]
        if missing:
            msg = (
                "HouyiDB server does not report the effective HNSW build parameters "
                f"({', '.join(missing)} missing from /v1/index/status); cannot verify that "
                f"m={self.case_config.m} and ef_construction={self.case_config.ef_construction} "
                "were applied. Upgrade HouyiDB, or set verify_build_params=False to accept "
                "the unverified configuration."
            )
            raise RuntimeError(msg)

        expected_m = int(self.case_config.m)
        expected_efc = int(self.case_config.ef_construction)
        actual_m = int(status["hnsw_m"])
        actual_efc = int(status["hnsw_ef_construction"])
        if actual_m != expected_m or actual_efc != expected_efc:
            hint = ""
            if self.transport == "pgwire":
                hint = f" Restart houyi-server with --pgwire-hnsw-m {expected_m} --pgwire-hnsw-efc {expected_efc}."
            msg = (
                f"HouyiDB effective HNSW parameters m={actual_m}, ef_construction={actual_efc} "
                f"do not match the case configuration m={expected_m}, ef_construction={expected_efc}."
                f"{hint}"
            )
            raise RuntimeError(msg)

        metric = status.get("distance_metric")
        if metric is not None and str(metric).lower() not in ("cosine", "cos"):
            msg = f"HouyiDB index metric is {metric!r}, expected cosine"
            raise RuntimeError(msg)

        self._build_params_verified = True

    def _execute_bounded(self, conn: Any, sql: str, params: Any) -> Any:
        """Run ``conn.execute`` under the client-side statement deadline."""
        self._pgwire_watchdog.start()
        token = self._pgwire_watchdog.register(conn, float(self.timeout_sec))
        try:
            result = conn.execute(sql, params)
        except Exception as exc:
            if token.timed_out.is_set():
                self._discard_pgwire_conn(conn)
                msg = f"HouyiDB pgwire query exceeded timeout_sec={self.timeout_sec}"
                raise TimeoutError(msg) from exc
            raise
        finally:
            self._pgwire_watchdog.unregister(token)
        if token.timed_out.is_set():
            self._discard_pgwire_conn(conn)
            msg = f"HouyiDB pgwire query exceeded timeout_sec={self.timeout_sec}"
            raise TimeoutError(msg)
        return result

    def _discard_pgwire_conn(self, conn: Any) -> None:
        """Drop a cancelled connection so the next call reconnects cleanly."""
        with self._pgwire_sessions_lock:
            if conn in self._pgwire_sessions:
                self._pgwire_sessions.remove(conn)
        if getattr(self._thread_local, "pg_conn", None) is conn:
            for attr in ("pg_conn", "pg_binary"):
                if hasattr(self._thread_local, attr):
                    delattr(self._thread_local, attr)
        with contextlib.suppress(Exception):
            # The connection is already unusable; closing is best-effort.
            conn.close()

    def _pgwire_conn(self) -> Any:
        if not hasattr(self._thread_local, "pg_conn"):
            try:
                import psycopg
                from psycopg.conninfo import make_conninfo
            except ImportError as exc:
                msg = "transport=pgwire requires psycopg; install 'psycopg[binary]>=3.1'"
                raise RuntimeError(msg) from exc
            params: dict[str, Any] = {
                "host": str(self.db_config["host"]),
                "port": str(self.db_config["port"]),
                "user": self._PGWIRE_USER,
                "dbname": self._PGWIRE_DATABASE,
                "connect_timeout": str(self.timeout_sec),
            }
            if self.api_key:
                params["password"] = str(self.api_key)
            conn = None
            try:
                conn = psycopg.connect(make_conninfo(**params), autocommit=True)
                self._execute_bounded(conn, "SELECT 1", None)
            except Exception as exc:
                if conn is not None:
                    with contextlib.suppress(Exception):
                        conn.close()
                msg = f"failed to connect HouyiDB pgwire {self.db_config['host']}:{self.db_config['port']}: {exc}"
                raise RuntimeError(msg) from exc
            with self._pgwire_sessions_lock:
                self._pgwire_sessions.append(conn)
            self._thread_local.pg_conn = conn
            self._thread_local.pg_binary = psycopg.Binary
        return self._thread_local.pg_conn

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self._connected = True
        if self.transport == "pgwire":
            self._pgwire_watchdog.start()
        try:
            if self.transport == "http":
                self._http_session()
            else:
                self._pgwire_conn()
            yield
        finally:
            self._connected = False
            self._pgwire_watchdog.stop()
            with self._http_sessions_lock:
                for session in self._http_sessions:
                    try:
                        session.close()
                    except Exception:
                        log.warning("failed to close HouyiDB HTTP session", exc_info=True)
                self._http_sessions.clear()
            if hasattr(self._thread_local, "session"):
                del self._thread_local.session
            if self.transport == "pgwire":
                with self._pgwire_sessions_lock:
                    for conn in self._pgwire_sessions:
                        try:
                            conn.close()
                        except Exception:
                            log.warning("failed to close HouyiDB pgwire connection", exc_info=True)
                    self._pgwire_sessions.clear()
                for attr in ("pg_conn", "pg_binary"):
                    if hasattr(self._thread_local, attr):
                        delattr(self._thread_local, attr)

    def optimize(self, data_size: int | None = None) -> None:
        """Trigger a server-side optimize and wait until the build backlog drains."""
        del data_size
        if not self._connected:
            msg = "Please call init() before optimize"
            raise RuntimeError(msg)
        self._verify_build_params()
        session = self._http_session()
        response = session.post(
            f"{self.base_url}/v1/optimize",
            timeout=(30, max(7200, self.timeout_sec)),
        )
        response.raise_for_status()
        poll_seconds = 0.05
        while True:
            response = session.get(f"{self.base_url}/v1/index/status", timeout=self.timeout_sec)
            response.raise_for_status()
            if int(response.json().get("pending_count", 0)) <= 0:
                return
            time.sleep(poll_seconds)

    def prepare_filter(self, filters: Filter) -> None:
        """Store the case filter; search_embedding applies it on the matching path."""
        if filters.type not in self.supported_filter_types:
            msg = f"unsupported HouyiDB filter: {filters}"
            raise ValueError(msg)
        self._filter = filters

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        del kwargs
        if not self._connected:
            return 0, RuntimeError("Please call init() before insert")
        if tenant_labels_data is not None:
            return 0, ValueError("HouyiDB does not support multitenant datasets")
        vectors = list(embeddings)
        if len(vectors) != len(metadata):
            return 0, ValueError("embeddings and metadata length mismatch")

        has_labels = labels_data is not None and len(labels_data) == len(vectors)
        if self.transport == "http" or has_labels:
            return self._insert_http(vectors, metadata, labels_data if has_labels else None)
        return self._insert_pgwire(vectors, metadata)

    def _insert_http(
        self,
        vectors: list[list[float]],
        metadata: list[int],
        labels: list[str] | None,
    ) -> tuple[int, Exception | None]:
        inserted = 0
        try:
            session = self._http_session()
            for start in range(0, len(vectors), self._HTTP_UPSERT_CHUNK_SIZE):
                end = start + self._HTTP_UPSERT_CHUNK_SIZE
                payload_vectors = []
                for idx, vector in enumerate(vectors[start:end], start=start):
                    item: dict[str, Any] = {"id": int(metadata[idx]), "values": vector}
                    if labels is not None:
                        item["label"] = labels[idx]
                    payload_vectors.append(item)
                payload = {
                    "vectors": payload_vectors,
                    "index": {
                        "m": int(self.case_config.m),
                        "ef_construction": int(self.case_config.ef_construction),
                    },
                }
                response = session.post(
                    f"{self.base_url}/v1/vectors/upsert",
                    json=payload,
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                inserted += int(response.json().get("inserted", 0))
        except Exception as exc:
            return inserted, exc
        return inserted, None

    def _insert_pgwire(
        self,
        vectors: list[list[float]],
        metadata: list[int],
    ) -> tuple[int, Exception | None]:
        inserted = 0
        try:
            conn = self._pgwire_conn()
            pg_binary = self._thread_local.pg_binary
            for start in range(0, len(vectors), self._HTTP_UPSERT_CHUNK_SIZE):
                end = min(start + self._HTTP_UPSERT_CHUNK_SIZE, len(vectors))
                chunk = []
                for idx in range(start, end):
                    vector = vectors[idx]
                    if len(vector) != self.dim:
                        return inserted, ValueError(f"dim mismatch, expected {self.dim}, got {len(vector)}")
                    chunk.append((int(metadata[idx]), vector))
                payload = _encode_upsert_batch(self.dim, chunk)
                row = self._execute_bounded(
                    conn,
                    "/* houyi:upsert_batch=%s */ SELECT 1",
                    (pg_binary(payload),),
                ).fetchone()
                inserted += int(row[0]) if row else 0
        except Exception as exc:
            return inserted, exc
        return inserted, None

    def _search_http(self, query: list[float], k: int) -> list[int]:
        payload: dict[str, Any] = {
            "vector": query,
            "k": int(k),
            "ef_search": int(self.case_config.ef_search),
        }
        if self._filter.type == FilterOp.NumGE and self._filter.int_value > 0:
            # HouyiDB's filter_gt_id is strict (>); VectorDBBench NumGE is >=.
            payload["filter_gt_id"] = int(self._filter.int_value) - 1
        elif self._filter.type == FilterOp.StrEqual:
            payload["filter_label"] = self._filter.label_value
        session = self._http_session()
        response = session.post(
            f"{self.base_url}/v1/vectors/search",
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return [int(item) for item in response.json().get("ids", [])]

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
        **kwargs: Any,
    ) -> list[int]:
        del payload_profile, tenant, kwargs
        if not self._connected:
            msg = "Please call init() before search"
            raise RuntimeError(msg)
        if len(query) != self.dim:
            msg = f"query dim mismatch, expected {self.dim}, got {len(query)}"
            raise ValueError(msg)
        self._verify_build_params()
        if self._filter.type != FilterOp.NonFilter:
            return self._search_http(query, k)
        if self.transport == "http":
            return self._search_http(query, k)
        conn = self._pgwire_conn()
        pg_binary = self._thread_local.pg_binary
        vector_payload = _encode_query_vector(query)
        ef_search = int(self.case_config.ef_search)
        sql = f"/* houyi:ef_search={ef_search} */ /* houyi:vector=%s */ SELECT id FROM docs LIMIT {int(k)}"
        rows = self._execute_bounded(conn, sql, (pg_binary(vector_payload),)).fetchall()
        return [int(row[0]) for row in rows]
