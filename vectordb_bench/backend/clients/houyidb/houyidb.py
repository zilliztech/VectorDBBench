"""HouyiDB client for VectorDBBench.

HouyiDB exposes two data planes:

* pgwire (PostgreSQL protocol) with a compact BYTEA batch format for vector
  upserts and queries;
* an HTTP benchmark API for bulk upserts, filtered search, health checks,
  optimize and index status.

This client uses pgwire for unfiltered workloads and the HTTP API when scalar
labels or id filters are required. Both paths target the same server-side
index. A fresh server (empty WAL) is required for a cold run because the
server has no client-side "drop collection" operation for the main index.
"""

from __future__ import annotations

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
        self.timeout_sec = int(db_config.get("timeout_sec", 600))
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
        self._filter: Filter = non_filter
        self._connected = False

        if drop_old:
            log.warning(
                "HouyiDB has no client-side drop for the main index; start a fresh "
                "server with an empty WAL for a cold run."
            )
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
            try:
                conn = psycopg.connect(make_conninfo(**params), autocommit=True)
                conn.execute("SELECT 1")
            except Exception as exc:
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
        try:
            if self.transport == "http":
                self._http_session()
            else:
                self._pgwire_conn()
            yield
        finally:
            self._connected = False
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
                row = conn.execute(
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
        try:
            session = self._http_session()
            response = session.post(
                f"{self.base_url}/v1/vectors/search",
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            return [int(item) for item in response.json().get("ids", [])]
        except Exception as exc:
            log.warning("HouyiDB HTTP search failed: %s", exc, exc_info=True)
            return []

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
        **kwargs: Any,
    ) -> list[int]:
        del payload_profile, tenant, kwargs
        if not self._connected or len(query) != self.dim:
            return []
        if self._filter.type != FilterOp.NonFilter:
            return self._search_http(query, k)
        if self.transport == "http":
            return self._search_http(query, k)
        try:
            conn = self._pgwire_conn()
            pg_binary = self._thread_local.pg_binary
            vector_payload = _encode_query_vector(query)
            ef_search = int(self.case_config.ef_search)
            sql = f"/* houyi:ef_search={ef_search} */ /* houyi:vector=%s */ SELECT id FROM docs LIMIT {int(k)}"
            rows = conn.execute(sql, (pg_binary(vector_payload),)).fetchall()
            return [int(row[0]) for row in rows]
        except Exception as exc:
            log.warning("HouyiDB pgwire search failed: %s", exc, exc_info=True)
            return []
