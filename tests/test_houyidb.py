"""Unit tests for the HouyiDB client.

These tests do not require a live server. They cover:
  - HouyiDBConfig / HouyiDBIndexConfig defaults and param assembly
  - AUB1 upsert and AVB1 query payload layout
  - DB registration (init_cls / config_cls / case_config_cls)
  - Filter routing: NumGE -> strict filter_gt_id, StrEqual -> filter_label
  - Scalar labels route inserts through the HTTP data plane
"""

from __future__ import annotations

import struct

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


def make_db(dim: int = 4, **overrides) -> HouyiDB:
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
        db_case_config=HouyiDBIndexConfig(metric_type=MetricType.COSINE),
    )


class _FakeResponse:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return {"ids": self._ids}


class _FakeSession:
    def __init__(self, ids: list[int] | None = None) -> None:
        self.captured: dict = {}
        self._ids = ids or []

    def post(self, url: str, json: dict, timeout: int) -> _FakeResponse:
        self.captured.update(url=url, json=json, timeout=timeout)
        return _FakeResponse(self._ids)


class TestHouyiDBConfig:
    def test_defaults_and_to_dict(self):
        cfg = HouyiDBConfig(host="db.local")
        assert cfg.port == 15432
        assert cfg.http_port == 17000
        assert cfg.transport == "pgwire"
        assert cfg.timeout_sec == 600
        assert cfg.to_dict() == {
            "host": "db.local",
            "port": 15432,
            "use_tls": False,
            "transport": "pgwire",
            "api_key": None,
            "timeout_sec": 600,
            "http_port": 17000,
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
