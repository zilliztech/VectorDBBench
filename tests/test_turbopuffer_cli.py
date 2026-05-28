from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.turbopuffer import cli as turbopuffer_cli
from vectordb_bench.backend.clients.turbopuffer import turbopuffer as turbopuffer_client


def test_turbopuffer_cli_accepts_multitenant_namespace_prefix_and_metric_type(
    monkeypatch: MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(turbopuffer_cli, "run", fake_run)

    result = CliRunner().invoke(
        turbopuffer_cli.TurboPuffer,
        [
            "--skip-drop-old",
            "--skip-load",
            "--skip-search-serial",
            "--search-concurrent",
            "--case-type",
            "CloudMultiTenantSearchCase",
            "--api-key",
            "secret",
            "--region",
            "aws-us-west-2",
            "--namespace",
            "cohere10m_multitenant",
            "--multitenant-namespace-prefix",
            "cohere10m_",
            "--scalar-payload-label-field",
            "scalar_label",
            "--metric-type",
            "COSINE",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured["db_config"].multitenant_namespace_prefix == "cohere10m_"
    assert captured["db_config"].scalar_payload_label_field == "scalar_label"
    assert captured["db_case_config"].metric_type == MetricType.COSINE


def test_turbopuffer_cli_skips_pin_namespace_during_dry_run(monkeypatch: MonkeyPatch) -> None:
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

    monkeypatch.setattr(turbopuffer_client, "namespace_metadata_request", fake_metadata_request)
    monkeypatch.setattr(turbopuffer_client, "wait_for_namespace_pinning", fake_wait_for_pinning)
    monkeypatch.setattr(turbopuffer_cli, "run", fake_run)

    result = CliRunner().invoke(
        turbopuffer_cli.TurboPuffer,
        [
            "--skip-drop-old",
            "--skip-load",
            "--skip-search-serial",
            "--search-concurrent",
            "--case-type",
            "CloudPayloadSearchCase",
            "--api-key",
            "secret",
            "--region",
            "aws-us-west-2",
            "--namespace",
            "laion100m",
            "--pin-namespace",
            "--pin-replicas",
            "2",
            "--pin-timeout",
            "7200",
            "--metric-type",
            "COSINE",
            "--disable-backpressure",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert captured["db_config"].pin_namespace is False
    assert captured["db_config"].pin_namespace_requested is True
    assert captured["db_config"].pin_replicas == 2
    assert captured["db_config"].pin_timeout == 7200
    assert captured["db_config"].pin_target_namespace_count == 1
    assert captured["db_case_config"].metric_type == MetricType.COSINE
    assert captured["db_case_config"].disable_backpressure is True


def test_turbopuffer_cli_skips_multitenant_pin_namespaces_during_dry_run(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = []
    captured = {}

    def fake_metadata_request(api_key, region, namespace, method, payload=None, api_base_url=None):
        calls.append((method, namespace, payload))
        return {}

    def fake_wait_for_pinning(api_key, region, namespace, replicas, api_base_url=None, timeout=None):
        calls.append(("WAIT", namespace, replicas))
        return {"pinning": {"replicas": replicas, "status": {"ready_replicas": replicas}}}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(turbopuffer_client, "namespace_metadata_request", fake_metadata_request)
    monkeypatch.setattr(turbopuffer_client, "wait_for_namespace_pinning", fake_wait_for_pinning)
    monkeypatch.setattr(turbopuffer_cli, "run", fake_run)

    result = CliRunner().invoke(
        turbopuffer_cli.TurboPuffer,
        [
            "--skip-drop-old",
            "--skip-load",
            "--skip-search-serial",
            "--search-concurrent",
            "--case-type",
            "CloudMultiTenantSearchCase",
            "--tenant-count",
            "2",
            "--tenant-prefix",
            "tenant_",
            "--tenant-id-width",
            "4",
            "--api-key",
            "secret",
            "--region",
            "aws-us-west-2",
            "--namespace",
            "unused_single_namespace",
            "--multitenant-namespace-prefix",
            "cohere10m_",
            "--pin-namespace",
            "--pin-replicas",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert captured["db_config"].pin_namespace is False
    assert captured["db_config"].pin_namespace_requested is True
    assert captured["db_config"].pin_target_namespace_count == 2
    assert captured["db_config"].multitenant_namespace_prefix == "cohere10m_"


def test_turbopuffer_unpin_namespace_uses_pin_timeout(monkeypatch: MonkeyPatch) -> None:
    calls = []

    def fake_metadata_request(api_key, region, namespace, method, payload=None, api_base_url=None):
        calls.append((method, payload, api_key, region, namespace, api_base_url))
        return {}

    def fake_wait_for_pinning(api_key, region, namespace, replicas, api_base_url=None, timeout=None):
        calls.append(("WAIT", replicas, api_key, region, namespace, api_base_url, timeout))
        return {"pinning": None}

    monkeypatch.setattr(turbopuffer_client, "namespace_metadata_request", fake_metadata_request)
    monkeypatch.setattr(turbopuffer_client, "wait_for_namespace_pinning", fake_wait_for_pinning)

    result = CliRunner().invoke(
        turbopuffer_cli.TurboPufferUnpin,
        [
            "--api-key",
            "secret",
            "--region",
            "aws-us-west-2",
            "--namespace",
            "laion100m",
            "--pin-timeout",
            "7200",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("PATCH", {"pinning": None}, "secret", "aws-us-west-2", "laion100m", None),
        ("WAIT", None, "secret", "aws-us-west-2", "laion100m", None, 7200),
    ]
