import base64
import json
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from click.testing import CliRunner

cbor2 = pytest.importorskip("cbor2")
elasticsearch = pytest.importorskip("elasticsearch")

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.aliyun_elasticsearch.aliyun_elasticsearch import AliyunElasticsearch
from vectordb_bench.backend.clients.aliyun_elasticsearch.config import (
    AliyunElasticsearchConfig,
    AliyunElasticsearchIndexConfig,
)
from vectordb_bench.cli import cli as common_cli
from vectordb_bench.cli.vectordbbench import cli
from vectordb_bench.models import CaseType, TaskStage


@pytest.fixture(autouse=True)
def clear_overrides(monkeypatch):
    for prefix in ("VDBBENCH_ES_", "VDBBENCH_ALIYUN_ES_"):
        for suffix in ("INDEX", "ID_FIELD", "QUERY_WIRE_FORMAT", "PASSWORD"):
            monkeypatch.delenv(prefix + suffix, raising=False)


@pytest.fixture
def http_server():
    state = {"requests": [], "status": 200, "require_auth": True}
    expected_auth = "Basic " + base64.b64encode(b"elastic:test-password").decode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"].append((self.path, self.headers, body))
            status = state["status"]
            if state["require_auth"] and self.headers.get("Authorization") != expected_auth:
                status = 401
            payload = (
                {"hits": {"hits": [{"fields": {"id": [7]}}]}}
                if status == 200
                else {"error": "authentication failed", "status": status}
            )
            response = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("X-Elastic-Product", "Elasticsearch")
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def make_client(url, **case_options):
    client = AliyunElasticsearch(
        dim=768,
        db_config={"hosts": [url], "basic_auth": ("elastic", "test-password"), "max_retries": 0},
        db_case_config=AliyunElasticsearchIndexConfig(
            query_wire_format="cbor-f32le",
            **case_options,
        ),
        indice="cohere10m_native_hnsw",
    )
    client.routing_key = None
    client.filter = []
    return client


@pytest.mark.parametrize(("k", "ef", "oversample"), [(10, 90, 4.6), (100, 408, 4.02)])
def test_authenticated_cbor_preserves_benchmark_request(http_server, k, ef, oversample):
    url, state = http_server
    client = make_client(url, num_candidates=ef, use_rescore=True, oversample_ratio=oversample)
    query = [(i % 37 - 18) / 19 for i in range(768)]
    with client.init():
        assert client.search_embedding(query, k=k) == [7]

    path, headers, body = state["requests"][0]
    assert path == "/cohere10m_native_hnsw/_search?filter_path=hits.hits.fields.id"
    assert headers["Content-Type"] == "application/cbor"
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"elastic:test-password").decode()
    # Same byte layout and field order as the frozen performance overlay.
    assert body == cbor2.dumps(
        {
            "knn": {
                "field": "vector",
                "k": k,
                "num_candidates": ef,
                "filter": [],
                "query_vector": struct.pack("<768f", *query),
                "rescore_vector": {"oversample": oversample},
            },
            "size": k,
            "_source": False,
            "docvalue_fields": ["id"],
            "stored_fields": "_none_",
        }
    )
    assert not hasattr(client, "client")
    assert not hasattr(client, "_cbor_headers")


def test_cbor_without_auth_and_rescore(http_server):
    url, state = http_server
    state["require_auth"] = False
    client = make_client(url, num_candidates=90)
    client.db_config.pop("basic_auth")
    with client.init():
        assert client.search_embedding([1.0], k=10) == [7]
    _, headers, body = state["requests"][0]
    assert "Authorization" not in headers
    assert "rescore_vector" not in cbor2.loads(body)["knn"]


def test_base64_transport_remains_big_endian_json(http_server):
    url, state = http_server
    client = make_client(url, num_candidates=90)
    client.query_wire_format = "base64-f32be"
    with client.init():
        try:
            assert client.search_embedding([1.0, -2.5], k=10) == [7]
        finally:
            client.client.close()
    _, headers, body = state["requests"][0]
    assert "json" in headers["Content-Type"]
    assert base64.b64decode(json.loads(body)["knn"]["query_vector"]) == struct.pack(">2f", 1.0, -2.5)


def test_cbor_reports_http_error_and_closes_client(http_server, monkeypatch):
    url, state = http_server
    state["status"] = 401
    client = make_client(url, num_candidates=90)
    closed = []
    with pytest.raises(elasticsearch.ApiError) as error:
        with client.init():
            close = client.client.close

            def record_close():
                closed.append(True)
                close()

            monkeypatch.setattr(client.client, "close", record_close)
            client.search_embedding([1.0], k=10)
    assert error.value.status_code == 401
    assert closed == [True]
    assert not hasattr(client, "client")


@pytest.mark.parametrize("index_name", [None, "configured-index"])
def test_index_setting_is_consumed_before_constructing_es_client(index_name, monkeypatch):
    created = []
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kwargs: created.append(kwargs))
    config = AliyunElasticsearchConfig(host="localhost", password="test-password", index_name=index_name)
    db_config = config.to_dict()
    client = AliyunElasticsearch(
        dim=768,
        db_config=db_config,
        db_case_config=AliyunElasticsearchIndexConfig(),
        indice="legacy-index",
    )
    assert client.indice == (index_name or "legacy-index")
    assert "index_name" not in created[0]
    assert db_config["index_name"] == index_name


def test_environment_overrides_keep_their_priority(monkeypatch):
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kwargs: None)
    monkeypatch.setenv("VDBBENCH_ES_INDEX", "generic-index")
    monkeypatch.setenv("VDBBENCH_ALIYUN_ES_INDEX", "aliyun-index")
    monkeypatch.setenv("VDBBENCH_ES_QUERY_WIRE_FORMAT", "base64-f32be")
    monkeypatch.setenv("VDBBENCH_ALIYUN_ES_QUERY_WIRE_FORMAT", "cbor-f32le")
    client = AliyunElasticsearch(
        dim=768,
        db_config={"index_name": "configured-index"},
        db_case_config=AliyunElasticsearchIndexConfig(),
    )
    assert client.indice == "aliyun-index"
    assert client.query_wire_format == "cbor-f32le"


def test_whitepaper_cli_builds_the_expected_task(monkeypatch):
    captured = []
    monkeypatch.setattr(common_cli.benchmark_runner, "run", lambda tasks, label: captured.extend(tasks))
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    monkeypatch.setenv("VDBBENCH_ES_PASSWORD", "test-password")
    result = CliRunner().invoke(
        cli,
        [
            "aliyunelasticsearch",
            "--scheme",
            "http",
            "--host",
            "localhost",
            "--port",
            "9200",
            "--user",
            "elastic",
            "--index-name",
            "cohere10m_native_hnsw",
            "--case-type",
            "Performance768D10M",
            "--skip-drop-old",
            "--skip-load",
            "--m",
            "32",
            "--ef-construction",
            "400",
            "--skip-search-serial",
            "--search-concurrent",
            "--concurrency-duration",
            "220",
            "--k",
            "10",
            "--num-candidates",
            "90",
            "--use-rescore",
            "--oversample-ratio",
            "4.6",
            "--num-concurrency",
            "212,216",
            "--db-label",
            "cohere10m-top10-dual16-c212-c216",
            "--query-wire-format",
            "cbor-f32le",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    (task,) = captured
    assert task.db == DB.AliyunElasticsearch
    assert task.db_config.index_name == "cohere10m_native_hnsw"
    assert task.db_config.password.get_secret_value() == "test-password"
    assert task.stages == [TaskStage.SEARCH_CONCURRENT]
    assert task.case_config.case_id == CaseType.Performance768D10M
    assert task.case_config.k == 10
    assert task.case_config.concurrency_search_config.num_concurrency == [212, 216]
    assert task.case_config.concurrency_search_config.concurrency_duration == 220
    assert task.db_case_config.query_wire_format == "cbor-f32le"
    assert task.db_case_config.M == 32
    assert task.db_case_config.efConstruction == 400
    assert task.db_case_config.num_candidates == 90
    assert task.db_case_config.use_rescore is True
    assert task.db_case_config.oversample_ratio == 4.6


def test_cli_keeps_base64_default(monkeypatch):
    captured = []
    monkeypatch.setattr(common_cli.benchmark_runner, "run", lambda tasks, label: captured.extend(tasks))
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    result = CliRunner().invoke(
        cli,
        [
            "aliyunelasticsearch",
            "--host",
            "localhost",
            "--password",
            "test-password",
            "--skip-drop-old",
            "--skip-load",
            "--skip-search-serial",
            "--search-concurrent",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured[0].db_case_config.query_wire_format == "base64-f32be"
