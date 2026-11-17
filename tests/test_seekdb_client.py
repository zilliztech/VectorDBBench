"""Offline unit tests for the SeekDB client setup path (no live SeekDB).

Covers what the web UI made reachable: reusing an existing index, the
instance-wide resource settings, credential logging and the HNSW pair check.
"""

import logging
import sys
import types

try:
    import mysql.connector  # noqa: F401
except ImportError:
    _mysql = types.ModuleType("mysql")
    _connector = types.ModuleType("mysql.connector")
    _connector.Error = type("Error", (Exception,), {})
    _connector.connect = None
    _mysql.connector = _connector
    sys.modules["mysql"] = _mysql
    sys.modules["mysql.connector"] = _connector

import pytest
from pydantic import ValidationError

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.seekdb import seekdb
from vectordb_bench.backend.clients.seekdb.config import SeekDBHNSWConfig

DB_CONFIG = {"host": "127.0.0.1", "port": 2881, "user": "root", "password": "synthetic-pw", "database": "bench"}

# SHOW CREATE TABLE output from SeekDB 1.4.0 for the index the client creates.
EXISTING_DDL = (
    "CREATE TABLE `items` (\n"
    "  `id` int(11) DEFAULT NULL,\n"
    "  `embedding` VECTOR(4) DEFAULT NULL,\n"
    "  VECTOR KEY `vidx` (`embedding`) WITH (DISTANCE=COSINE, TYPE=HNSW, M=16, EF_CONSTRUCTION=256, "
    "LIB=VSAG, EF_SEARCH=64, SYNC_MODE=ASYNC) BLOCK_SIZE 16384\n"
    ") ORGANIZATION HEAP DEFAULT CHARSET = utf8mb4"
)


class FakeCursor:
    def __init__(self, ddl: str):
        self.ddl = ddl
        self.queries = []

    def execute(self, query):
        self.queries.append(query)

    def fetchone(self):
        return ("items", self.ddl)

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


@pytest.fixture
def cursor(monkeypatch):
    fake = FakeCursor(EXISTING_DDL)
    monkeypatch.setattr(seekdb.mysql, "connect", lambda **_: FakeConnection(fake), raising=False)
    return fake


def _case_config(**kwargs) -> SeekDBHNSWConfig:
    params = {"m": 16, "ef_construction": 256, "ef_search": 100, "metric_type": MetricType.COSINE} | kwargs
    return SeekDBHNSWConfig(**params)


def test_reuse_accepts_matching_index(cursor):
    seekdb.SeekDB(4, DB_CONFIG, _case_config(), drop_old=False)

    assert cursor.queries == ["SHOW CREATE TABLE items"]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"m": 32}, "'m'"),
        ({"ef_construction": 200}, "'ef_construction'"),
        ({"metric_type": MetricType.L2}, "'distance'"),
    ],
)
@pytest.mark.usefixtures("cursor")
def test_reuse_rejects_mismatched_index(kwargs, field):
    with pytest.raises(ValueError, match=field):
        seekdb.SeekDB(4, DB_CONFIG, _case_config(**kwargs), drop_old=False)


def test_reuse_rejects_unverifiable_index(cursor):
    cursor.ddl = "CREATE TABLE `items` (\n  `id` int(11) DEFAULT NULL\n) ORGANIZATION HEAP"

    with pytest.raises(ValueError, match="Cannot verify"):
        seekdb.SeekDB(4, DB_CONFIG, _case_config(), drop_old=False)


def test_resource_limits_untouched_by_default(cursor):
    seekdb.SeekDB(4, DB_CONFIG, _case_config(), drop_old=True)

    assert not any("ALTER SYSTEM" in q for q in cursor.queries)


def test_resource_limits_reset_when_opted_in(cursor):
    seekdb.SeekDB(4, DB_CONFIG, _case_config(reset_resource_limits=True), drop_old=True)

    assert 'ALTER SYSTEM SET memory_limit = "0M"' in cursor.queries
    assert "ALTER SYSTEM SET cpu_count = 0" in cursor.queries


@pytest.mark.usefixtures("cursor")
def test_password_is_not_logged(caplog):
    with caplog.at_level(logging.INFO, logger=seekdb.log.name):
        seekdb.SeekDB(4, DB_CONFIG, _case_config(), drop_old=True)

    assert "SeekDB initialized" in caplog.text
    assert "synthetic-pw" not in caplog.text


def test_ef_construction_must_exceed_m():
    with pytest.raises(ValidationError, match="ef_construction > m"):
        _case_config(m=16, ef_construction=16)

    assert _case_config(m=16, ef_construction=17).ef_construction == 17
