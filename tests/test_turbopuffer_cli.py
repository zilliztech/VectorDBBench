from click.testing import CliRunner

from vectordb_bench.backend.clients.turbopuffer import cli as tpuf_cli
from vectordb_bench.backend.clients.turbopuffer import turbopuffer as tpuf_client


def test_pin_namespace_is_applied_once_before_benchmark(monkeypatch):
    calls = []
    captured = {}

    def fake_metadata_request(api_key, region, namespace, method, payload=None, api_base_url=None):
        calls.append((method, payload, api_key, region, namespace, api_base_url))
        return {}

    def fake_wait_for_pinning(api_key, region, namespace, replicas, api_base_url=None, timeout=None):
        calls.append(("WAIT", replicas, api_key, region, namespace, api_base_url, timeout))
        return {"pinning": {"replicas": replicas, "status": {"ready_replicas": replicas}}}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tpuf_client, "namespace_metadata_request", fake_metadata_request)
    monkeypatch.setattr(tpuf_client, "wait_for_namespace_pinning", fake_wait_for_pinning)
    monkeypatch.setattr(tpuf_cli, "run", fake_run)

    result = CliRunner().invoke(
        tpuf_cli.TurboPuffer,
        [
            "--api-key",
            "test-key",
            "--region",
            "aws-us-west-2",
            "--namespace",
            "test-ns",
            "--pin-namespace",
            "--pin-replicas",
            "1",
            "--pin-timeout",
            "7200",
            "--case-type",
            "Performance768D100M",
            "--skip-drop-old",
            "--skip-load",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("PATCH", {"pinning": {"replicas": 1}}, "test-key", "aws-us-west-2", "test-ns", None),
        ("WAIT", 1, "test-key", "aws-us-west-2", "test-ns", None, 7200),
    ]
    assert captured["db_config"].pin_namespace is False
    assert captured["db_config"].pin_replicas == 1


def test_unpin_namespace_uses_pin_timeout(monkeypatch):
    calls = []

    def fake_metadata_request(api_key, region, namespace, method, payload=None, api_base_url=None):
        calls.append((method, payload, api_key, region, namespace, api_base_url))
        return {}

    def fake_wait_for_pinning(api_key, region, namespace, replicas, api_base_url=None, timeout=None):
        calls.append(("WAIT", replicas, api_key, region, namespace, api_base_url, timeout))
        return {"pinning": None}

    monkeypatch.setattr(tpuf_client, "namespace_metadata_request", fake_metadata_request)
    monkeypatch.setattr(tpuf_client, "wait_for_namespace_pinning", fake_wait_for_pinning)

    result = CliRunner().invoke(
        tpuf_cli.TurboPufferUnpin,
        [
            "--api-key",
            "test-key",
            "--region",
            "aws-us-west-2",
            "--namespace",
            "test-ns",
            "--pin-timeout",
            "7200",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("PATCH", {"pinning": None}, "test-key", "aws-us-west-2", "test-ns", None),
        ("WAIT", None, "test-key", "aws-us-west-2", "test-ns", None, 7200),
    ]
