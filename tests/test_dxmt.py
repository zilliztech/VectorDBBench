"""Unit tests for the DXMT client. No server needed: requests go to an httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from click.testing import CliRunner

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.dxmt import cli as dxmt_cli
from vectordb_bench.backend.clients.dxmt import dxmt as dxmt_module
from vectordb_bench.backend.clients.dxmt.config import DXMTAutoIndexConfig, DXMTConfig, DXMTHNSWConfig
from vectordb_bench.backend.clients.dxmt.dxmt import DXMT, DXMTError
from vectordb_bench.backend.filter import IntFilter, LabelFilter, non_filter


def ok(data: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json={"success": True, "data": data or {}, "error": None})


def err(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"success": False, "data": None, "error": {"code": code, "message": code}})


class FakeServer:
    """Records every request and answers like the DXMT REST API."""

    def __init__(self, table_exists: bool = False):
        self.requests: list[httpx.Request] = []
        self.table_exists = table_exists
        self.created_hnsw_override = None
        self.overrides: dict[tuple[str, str], list[httpx.Response]] = {}

    def queue(self, method: str, suffix: str, *responses: httpx.Response):
        self.overrides.setdefault((method, suffix), []).extend(responses)

    def bodies(self, method: str, suffix: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == method and r.url.path.endswith(suffix)]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for (method, suffix), queued in self.overrides.items():
            if request.method == method and path.endswith(suffix) and queued:
                return queued.pop(0)
        if request.method == "GET" and path.startswith("/api/v1/tables/"):
            return ok({"name": "t"}) if self.table_exists else err(404, "TABLE_NOT_FOUND")
        if request.method == "DELETE":
            if not self.table_exists:
                return err(404, "TABLE_NOT_FOUND")
            self.table_exists = False
            return ok()
        if request.method == "POST" and path == "/api/v1/tables":
            body = json.loads(request.content)
            self.table_exists = True
            return ok({"name": body["name"], "hnsw": self.created_hnsw_override or body.get("hnsw")})
        if path.endswith("/records"):
            n = len(json.loads(request.content)["records"])
            return ok({"count": n, "failed": [], "failed_count": 0})
        if path.endswith("/search"):
            return ok({"hits": [{"id": "7", "score": 0.9}, {"id": "3", "score": 0.8}]})
        if path.endswith("/index/rebuild"):
            return ok({"table": "t"})
        return err(400, "UNEXPECTED")


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    fake = FakeServer()

    def connect(self):
        return httpx.Client(base_url=self.base_url, headers=self.headers, transport=httpx.MockTransport(fake))

    monkeypatch.setattr(DXMT, "_connect", connect)
    return fake


def make_db(case_config=None, **kwargs) -> DXMT:
    return DXMT(
        dim=4,
        db_config=DXMTConfig(db_label="t", url="http://dxmt:9999/").to_dict(),
        db_case_config=case_config or DXMTAutoIndexConfig(metric_type=MetricType.COSINE),
        collection_name="bench",
        **kwargs,
    )


def test_registration():
    assert DB.DXMT.init_cls is DXMT
    assert DB.DXMT.config_cls is DXMTConfig
    assert DB.DXMT.case_config_cls(IndexType.HNSW) is DXMTHNSWConfig
    assert DB.DXMT.case_config_cls(IndexType.AUTOINDEX) is DXMTAutoIndexConfig
    assert DB.DXMT.case_config_cls() is DXMTAutoIndexConfig


def test_index_params():
    assert DXMTAutoIndexConfig(metric_type=MetricType.L2).index_param() == {"distance_metric": "euclidean"}
    hnsw = DXMTHNSWConfig(metric_type=MetricType.IP, m=32, ef_construction=256, ef_search=64)
    assert hnsw.index_param() == {
        "distance_metric": "dot",
        "index_type": "hnsw",
        "hnsw": {"m": 32, "ef_construction": 256, "ef_search": 64},
    }
    assert hnsw.search_param() == {}
    assert DXMTAutoIndexConfig(search_mode="accurate").search_param() == {"search_mode": "accurate"}


def test_api_key_is_sent(server: FakeServer):
    DXMT(
        dim=4,
        db_config=DXMTConfig(db_label="t", api_key="secret").to_dict(),
        db_case_config=DXMTAutoIndexConfig(),
    )
    assert all(r.headers["api-key"] == "secret" for r in server.requests)


def test_creates_table_with_server_defaults(server: FakeServer):
    make_db()
    assert server.bodies("POST", "/tables") == [{"name": "bench", "dimension": 4, "distance_metric": "cosine"}]


def test_drop_old_drops_then_creates(server: FakeServer):
    server.table_exists = True
    make_db(drop_old=True)
    methods = [(r.method, r.url.path) for r in server.requests]
    assert methods == [
        ("DELETE", "/api/v1/tables/bench"),
        ("GET", "/api/v1/tables/bench"),
        ("POST", "/api/v1/tables"),
    ]


def test_existing_table_is_kept_without_drop_old(server: FakeServer):
    server.table_exists = True
    make_db()
    assert [r.method for r in server.requests] == ["GET"]


def test_hnsw_params_the_server_did_not_apply_abort(server: FakeServer):
    server.created_hnsw_override = {"m": 16, "ef_construction": 100, "ef_search": 128}
    with pytest.raises(DXMTError, match="requested"):
        make_db(DXMTHNSWConfig(metric_type=MetricType.COSINE, ef_construction=200))


def test_insert_writes_id_and_label_payload(server: FakeServer, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dxmt_module, "MAX_RECORDS_PER_REQUEST", 2)
    db = make_db(with_scalar_labels=True)
    vectors = list(np.arange(12, dtype=np.float32).reshape(3, 4))
    with db.init():
        count, error = db.insert_embeddings(vectors, [10, 11, 12], labels_data=["a", "b", "c"])
    assert (count, error) == (3, None)
    batches = server.bodies("POST", "/records")
    assert [len(b["records"]) for b in batches] == [2, 1]
    assert batches[1]["records"][0] == {
        "id": "12",
        "vector": [8.0, 9.0, 10.0, 11.0],
        "payload": {"id_num": 12, "label": "c"},
    }


def test_insert_without_labels_writes_only_the_id(server: FakeServer):
    db = make_db()
    with db.init():
        db.insert_embeddings([[0.1, 0.2, 0.3, 0.4]], [5], labels_data=["ignored"])
    assert server.bodies("POST", "/records")[0]["records"][0]["payload"] == {"id_num": 5}


def test_insert_waits_out_backpressure(server: FakeServer, monkeypatch: pytest.MonkeyPatch):
    sleeps = []
    monkeypatch.setattr(dxmt_module.time, "sleep", sleeps.append)
    server.queue("POST", "/records", httpx.Response(429, headers={"retry-after": "2"}, json={"success": False}))
    db = make_db()
    with db.init():
        assert db.insert_embeddings([[0.0] * 4], [1]) == (1, None)
    assert sleeps == [2.0]


def test_insert_reports_rejected_rows(server: FakeServer):
    server.queue("POST", "/records", ok({"count": 1, "failed": [{"index": 1}], "failed_count": 1}))
    db = make_db()
    with db.init():
        count, error = db.insert_embeddings([[0.0] * 4, [1.0] * 4], [1, 2])
    assert count == 0
    assert isinstance(error, DXMTError)


def test_search_body_and_filters(server: FakeServer):
    db = make_db(DXMTAutoIndexConfig(metric_type=MetricType.COSINE, search_mode="accurate"))
    with db.init():
        db.prepare_filter(non_filter)
        assert db.search_embedding(np.zeros(4, dtype=np.float32), k=2) == [7, 3]
        db.prepare_filter(IntFilter(filter_rate=0.99, int_value=990))
        db.search_embedding([0.0] * 4, k=2)
        db.prepare_filter(LabelFilter(label_percentage=0.01))
        db.search_embedding([0.0] * 4, k=2)
    plain, num_ge, label = server.bodies("POST", "/search")
    assert plain == {"vector": [0.0] * 4, "top_k": 2, "with_payload": False, "search_mode": "accurate"}
    assert num_ge["filter"] == {"must": [{"key": "id_num", "operation": {"type": "GreaterThanOrEqual", "value": 990}}]}
    assert label["filter"] == {"must": [{"key": "label", "operation": {"type": "Equal", "value": "label_1p"}}]}


def test_search_failure_raises(server: FakeServer):
    server.queue("POST", "/search", err(500, "INTERNAL"))
    db = make_db()
    with db.init(), pytest.raises(DXMTError, match="INTERNAL"):
        db.search_embedding([0.0] * 4)


def test_optimize_rebuilds_and_raises_on_failure(server: FakeServer):
    db = make_db()
    db.optimize()
    assert server.requests[-1].url.path == "/api/v1/tables/bench/index/rebuild"
    server.queue("POST", "/index/rebuild", err(503, "INSUFFICIENT_STORAGE"))
    with pytest.raises(DXMTError):
        db.optimize()


@pytest.mark.parametrize(
    ("command", "extra", "config_cls"),
    [
        (dxmt_cli.DXMT, [], DXMTAutoIndexConfig),
        (dxmt_cli.DXMTHNSW, ["--m", "32", "--ef-construction", "256", "--ef-search", "64"], DXMTHNSWConfig),
    ],
)
def test_cli(monkeypatch: pytest.MonkeyPatch, command, extra, config_cls):
    captured = {}
    monkeypatch.setattr(dxmt_cli, "run", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        command,
        ["--url", "http://h:1", "--search-mode", "FAST", "--case-type", "Performance768D1M", *extra],
    )
    assert result.exit_code == 0, result.output
    assert captured["db"] == DB.DXMT
    assert captured["db_config"].url == "http://h:1"
    assert captured["db_config"].api_key is None
    assert isinstance(captured["db_case_config"], config_cls)
    assert captured["db_case_config"].search_mode == "fast"
    if config_cls is DXMTHNSWConfig:
        assert captured["db_case_config"].index_param()["hnsw"] == {"m": 32, "ef_construction": 256, "ef_search": 64}
