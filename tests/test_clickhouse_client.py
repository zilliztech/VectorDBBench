"""Offline unit tests for the Clickhouse search query (no live ClickHouse)."""

import sys
import types

try:
    import clickhouse_connect  # noqa: F401
except ImportError:
    _cc = types.ModuleType("clickhouse_connect")
    _driver = types.ModuleType("clickhouse_connect.driver")
    _driver.Client = object
    _cc.driver = _driver
    _cc.get_client = None
    sys.modules["clickhouse_connect"] = _cc
    sys.modules["clickhouse_connect.driver"] = _driver

import pytest

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.clickhouse.clickhouse import Clickhouse
from vectordb_bench.backend.clients.clickhouse.config import ClickhouseHNSWConfig

DB_CONFIG = {"host": "127.0.0.1", "port": 8123, "user": "clickhouse", "password": "pw", "database": "default"}


class FakeConn:
    def __init__(self):
        self.queries = []

    def close(self):
        pass

    def query(self, sql, parameters=None, settings=None):
        self.queries.append((sql, settings))
        return types.SimpleNamespace(result_rows=[(3,), (1,)])


@pytest.fixture(autouse=True)
def _no_server(monkeypatch):
    monkeypatch.setattr(Clickhouse, "_create_connection", lambda self, settings, **kwargs: FakeConn())


def _client(ef, metric_type=MetricType.L2):
    config = ClickhouseHNSWConfig(M=16, efConstruction=128, ef=ef, metric_type=metric_type)
    client = Clickhouse(dim=4, db_config=DB_CONFIG, db_case_config=config)
    client.conn = FakeConn()
    return client


def test_ef_is_sent_as_the_hnsw_search_setting():
    client = _client(ef=64)

    assert client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10) == [3, 1]
    _, settings = client.conn.queries[0]
    assert settings == {"hnsw_candidate_list_size_for_search": 64}


def test_no_ef_keeps_the_server_default():
    client = _client(ef=None)

    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    _, settings = client.conn.queries[0]
    assert settings is None


@pytest.mark.parametrize(
    ("metric_type", "filters", "distance", "has_where"),
    [
        (MetricType.L2, None, "L2Distance", False),
        (MetricType.L2, {"id": 100}, "L2Distance", True),
        (MetricType.COSINE, None, "cosineDistance", False),
        (MetricType.COSINE, {"id": 100}, "cosineDistance", True),
    ],
)
def test_search_selects_only_ids(metric_type, filters, distance, has_where):
    client = _client(ef=64, metric_type=metric_type)

    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10, filters=filters)
    sql, _ = client.conn.queries[0]
    assert sql.startswith("SELECT {primary_field:Identifier} FROM")
    assert "{vector_field:Identifier}" not in sql
    assert f"ORDER BY {distance}(embedding,{{query:Array(Float64)}})" in sql
    assert ("WHERE" in sql) == has_where
