from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.turbopuffer import cli as turbopuffer_cli


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
            "--metric-type",
            "COSINE",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured["db_config"].multitenant_namespace_prefix == "cohere10m_"
    assert captured["db_case_config"].metric_type == MetricType.COSINE
