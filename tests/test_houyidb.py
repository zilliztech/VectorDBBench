"""Unit tests for the HouyiDB client.

These tests do not require a live server. They cover:
  - HouyiDBConfig / HouyiDBIndexConfig defaults and param assembly
  - AUB1 upsert and AVB1 query payload layout
  - DB registration (init_cls / config_cls / case_config_cls)
  - Filter routing: NumGE -> strict filter_gt_id, StrEqual -> filter_label
  - Scalar labels route inserts through the HTTP data plane
  - Rejection of non-cosine datasets
  - drop_old enforcement against an occupied index
  - Effective HNSW build-parameter verification
  - Search failure propagation on both transports
  - pgwire statement timeout
"""

from __future__ import annotations

import struct
import threading
import time

import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.houyidb.config import HouyiDBConfig, HouyiDBIndexConfig
from vectordb_bench.backend.clients.houyidb.houyidb import (
    HouyiDB,
    _encode_query_vector,
    _encode_upsert_batch,
)
from vectordb_bench.backend.filter import IntFilter, LabelFilter


def make_db(
    dim: int = 4,
    *,
    case_config: HouyiDBIndexConfig | None = None,
    drop_old: bool = False,
    **overrides,
) -> HouyiDB:
    db_config = {
        "host": "127.0.0.1",
        "port": 15432,
        "http_port": 17000,
        "transport": "pgwire",
    }
    db_config.update(overrides)
    return HouyiDB(
        dim=dim,
        db_config=db_config,
        db_case_config=case_config or HouyiDBIndexConfig(metric_type=MetricType.COSINE),
        drop_old=drop_old,
    )


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise RuntimeError(msg)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(
        self,
        ids: list[int] | None = None,
        status: dict | None = None,
        post_error: Exception | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.captured: dict = {}
        self.posts: list[dict] = []
        self.gets: list[str] = []
        self._ids = ids or []
        self._status = status if status is not None else self.default_status()
        self._post_error = post_error
        self._get_error = get_error

    @staticmethod
    def default_status(**overrides) -> dict:
        status = {
            "hot_count": 0,
            "indexed_count": 0,
            "pending_count": 0,
            "hnsw_m": 24,
            "hnsw_ef_construction": 400,
            "distance_metric": "cosine",
        }
        status.update(overrides)
        return status

    def get(self, url: str, timeout: int | None = None) -> _FakeResponse:
        self.gets.append(url)
        if self._get_error is not None:
            raise self._get_error
        if url.endswith("/v1/index/status"):
            return _FakeResponse(self._status)
        if url.endswith("/v1/health"):
            return _FakeResponse({"status": "ok"})
        return _FakeResponse({})

    def post(self, url: str, json: dict | None = None, timeout: int | None = None) -> _FakeResponse:
        if self._post_error is not None:
            raise self._post_error
        self.captured.update(url=url, json=json, timeout=timeout)
        self.posts.append({"url": url, "json": json})
        if url.endswith("/v1/vectors/search"):
            return _FakeResponse({"ids": self._ids})
        if url.endswith("/v1/optimize"):
            return _FakeResponse({"pending_items": 0})
        return _FakeResponse({"inserted": len((json or {}).get("vectors", []))})


class _BrokenConn:
    def execute(self, sql: str, params: object) -> None:
        raise RuntimeError("server exploded")

    def close(self) -> None:
        pass


class _BlockingConn:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.closed = False

    def execute(self, sql: str, params: object) -> None:
        self.cancelled.wait(timeout=5)
        raise RuntimeError("query cancelled")

    def cancel_safe(self, timeout: float = 5) -> None:
        self.cancelled.set()

    def close(self) -> None:
        self.closed = True
        self.cancelled.set()


def _patch_http(monkeypatch: pytest.MonkeyPatch, db: HouyiDB, session: _FakeSession) -> _FakeSession:
    db._connected = True
    monkeypatch.setattr(db, "_http_session", lambda: session)
    return session


class TestHouyiDBConfig:
    def test_defaults_and_to_dict(self):
        cfg = HouyiDBConfig(host="db.local")
        assert cfg.port == 15432
        assert cfg.http_port == 17000
        assert cfg.transport == "pgwire"
        assert cfg.timeout_sec == 600
        assert cfg.verify_build_params is True
        assert cfg.to_dict() == {
            "host": "db.local",
            "port": 15432,
            "use_tls": False,
            "transport": "pgwire",
            "api_key": None,
            "timeout_sec": 600,
            "http_port": 17000,
            "verify_build_params": True,
        }

    def test_index_and_search_param(self):
        cfg = HouyiDBIndexConfig(
            metric_type=MetricType.L2,
            m=16,
            ef_construction=200,
            ef_search=64,
        )
        assert cfg.index_param() == {
            "metric_type": "L2",
            "m": 16,
            "ef_construction": 200,
        }
        assert cfg.search_param() == {"ef_search": 64}


class TestEncoders:
    def test_upsert_batch_layout(self):
        payload = _encode_upsert_batch(3, [(7, [1.0, 2.0, 3.0])])
        assert payload[:4] == b"AUB1"
        dim, count = struct.unpack("<II", payload[4:12])
        assert (dim, count) == (3, 1)
        (vector_id,) = struct.unpack("<Q", payload[12:20])
        assert vector_id == 7
        assert struct.unpack("<fff", payload[20:32]) == (1.0, 2.0, 3.0)

    def test_upsert_batch_rejects_dim_mismatch(self):
        with pytest.raises(ValueError, match="dim mismatch"):
            _encode_upsert_batch(3, [(1, [1.0, 2.0])])

    def test_query_vector_layout(self):
        payload = _encode_query_vector([0.5, 0.25])
        assert payload[:4] == b"AVB1"
        (dim,) = struct.unpack("<I", payload[4:8])
        assert dim == 2
        assert struct.unpack("<ff", payload[8:]) == (0.5, 0.25)


class TestRegistration:
    def test_db_resolution(self):
        assert DB.HouyiDB.init_cls is HouyiDB
        assert DB.HouyiDB.config_cls is HouyiDBConfig
        assert DB.HouyiDB.case_config_cls() is HouyiDBIndexConfig

    def test_http_port_falls_back_to_pgwire_port(self):
        db = make_db(http_port=None, transport="http")
        assert db.base_url == "http://127.0.0.1:15432"


class TestMetricSupport:
    @pytest.mark.parametrize("metric", [MetricType.L2, MetricType.IP])
    def test_rejects_non_cosine_datasets(self, metric: MetricType):
        with pytest.raises(ValueError, match="cosine index"):
            make_db(case_config=HouyiDBIndexConfig(metric_type=metric))


class TestDropOld:
    def test_occupied_index_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        session = _FakeSession(status=_FakeSession.default_status(hot_count=1, indexed_count=2))
        monkeypatch.setattr(HouyiDB, "_http_session", lambda *_: session)
        with pytest.raises(RuntimeError, match="not empty"):
            make_db(drop_old=True)

    def test_empty_index_is_accepted(self, monkeypatch: pytest.MonkeyPatch):
        session = _FakeSession()
        monkeypatch.setattr(HouyiDB, "_http_session", lambda *_: session)
        db = make_db(drop_old=True)
        assert db.metric is MetricType.COSINE
        assert any(url.endswith("/v1/index/status") for url in session.gets)


class TestBuildParamVerification:
    def test_mismatch_is_rejected_before_optimize(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        session = _patch_http(
            monkeypatch,
            db,
            _FakeSession(status=_FakeSession.default_status(hnsw_m=16)),
        )
        with pytest.raises(RuntimeError, match="do not match"):
            db.optimize()
        assert not any(post["url"].endswith("/v1/optimize") for post in session.posts)

    def test_pgwire_mismatch_mentions_startup_flags(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        _patch_http(
            monkeypatch,
            db,
            _FakeSession(status=_FakeSession.default_status(hnsw_ef_construction=128)),
        )
        with pytest.raises(RuntimeError, match="--pgwire-hnsw-efc 400"):
            db.optimize()

    def test_missing_fields_are_rejected(self, monkeypatch: pytest.MonkeyPatch):
        status = _FakeSession.default_status()
        status.pop("hnsw_m")
        db = make_db()
        _patch_http(monkeypatch, db, _FakeSession(status=status))
        with pytest.raises(RuntimeError, match="does not report"):
            db.optimize()

    def test_verification_can_be_disabled_for_older_servers(self, monkeypatch: pytest.MonkeyPatch):
        status = _FakeSession.default_status()
        status.pop("hnsw_m")
        db = make_db(verify_build_params=False)
        session = _patch_http(monkeypatch, db, _FakeSession(status=status))
        db.optimize()
        assert any(post["url"].endswith("/v1/optimize") for post in session.posts)

    def test_matching_params_allow_optimize(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        session = _patch_http(monkeypatch, db, _FakeSession())
        db.optimize()
        assert any(post["url"].endswith("/v1/optimize") for post in session.posts)


class TestSearchFailures:
    def test_http_failure_propagates(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db(transport="http")
        _patch_http(monkeypatch, db, _FakeSession(post_error=RuntimeError("connection reset")))
        with pytest.raises(RuntimeError, match="connection reset"):
            db.search_embedding([0.0, 0.0, 0.0, 0.0], k=2)

    def test_pgwire_failure_propagates(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        db._connected = True
        db._build_params_verified = True
        db._thread_local.pg_binary = lambda payload: payload
        monkeypatch.setattr(db, "_pgwire_conn", _BrokenConn)
        with pytest.raises(RuntimeError, match="server exploded"):
            db.search_embedding([0.0, 0.0, 0.0, 0.0], k=2)

    def test_successful_zero_hits_return_empty(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db(transport="http")
        _patch_http(monkeypatch, db, _FakeSession(ids=[]))
        assert db.search_embedding([0.0, 0.0, 0.0, 0.0], k=2) == []

    def test_search_before_init_raises(self):
        db = make_db()
        with pytest.raises(RuntimeError, match="init"):
            db.search_embedding([0.0, 0.0, 0.0, 0.0], k=2)

    def test_query_dim_mismatch_raises(self):
        db = make_db()
        db._connected = True
        with pytest.raises(ValueError, match="dim mismatch"):
            db.search_embedding([0.0, 0.0], k=2)


class TestPgwireTimeout:
    def test_blocked_statement_times_out_and_closes(self):
        db = make_db()
        db.timeout_sec = 1
        conn = _BlockingConn()
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="timeout_sec=1"):
                db._execute_bounded(conn, "SELECT 1", None)
        finally:
            db._pgwire_watchdog.stop()
        assert time.monotonic() - started < 3.0
        assert conn.closed


class TestFilterRouting:
    def test_num_ge_uses_strict_gt_id(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        db._connected = True
        db.prepare_filter(IntFilter(int_value=100))
        session = _FakeSession(ids=[3, 5])
        monkeypatch.setattr(db, "_http_session", lambda: session)

        ids = db.search_embedding([0.1, 0.2, 0.3, 0.4], k=5)

        assert ids == [3, 5]
        assert session.captured["json"]["filter_gt_id"] == 99
        assert "filter_label" not in session.captured["json"]

    def test_label_filter_uses_label_payload(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        db._connected = True
        db.prepare_filter(LabelFilter(label_percentage=0.01))
        session = _FakeSession(ids=[1])
        monkeypatch.setattr(db, "_http_session", lambda: session)

        db.search_embedding([0.0, 0.0, 0.0, 0.0], k=10)

        assert session.captured["json"]["filter_label"] == "label_1p"
        assert "filter_gt_id" not in session.captured["json"]

    def test_scalar_labels_force_http_upsert(self, monkeypatch: pytest.MonkeyPatch):
        db = make_db()
        db._connected = True
        captured: dict = {}

        def fake_insert_http(
            vectors: list[list[float]],
            metadata: list[int],
            labels: list[str] | None,
        ) -> tuple[int, None]:
            captured.update(vectors=vectors, metadata=metadata, labels=labels)
            return len(vectors), None

        monkeypatch.setattr(db, "_insert_http", fake_insert_http)
        inserted, err = db.insert_embeddings(
            [[0.0, 0.0, 0.0, 0.0]],
            [42],
            labels_data=["label_1p"],
        )

        assert err is None
        assert inserted == 1
        assert captured["metadata"] == [42]
        assert captured["labels"] == ["label_1p"]
