import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from pytest import MonkeyPatch

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.test import cli as test_cli
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.cli import cli as common_cli
from vectordb_bench.models import CaseConfig


def invoke_test_command(monkeypatch: MonkeyPatch, args: list[str]):
    captured = {}

    def fake_run(tasks, task_label):
        captured["task"] = tasks[0]
        captured["task_label"] = task_label

    monkeypatch.setattr(common_cli.benchmark_runner, "run", fake_run)
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    result = CliRunner().invoke(test_cli.Test, args)
    return result, captured


def test_case_config_rejects_non_positive_k():
    with pytest.raises(ValidationError, match="positive"):
        CaseConfig(case_id=CaseType.Performance768D100M, k=0)


def test_cli_rejects_non_positive_k():
    result = CliRunner().invoke(test_cli.Test, ["--k", "0", "--dry-run"])

    assert result.exit_code != 0
    assert "range" in result.output


def test_cli_help_describes_laion_large_topk_limit():
    result = CliRunner().invoke(test_cli.Test, ["--help"])

    assert result.exit_code == 0, result.output
    assert "LAION" in result.output
    assert "1,000,000" in result.output


def test_cli_builds_laion_large_topk_integer_filter_case(monkeypatch: MonkeyPatch):
    result, captured = invoke_test_command(
        monkeypatch,
        [
            "--case-type",
            "NewIntFilterPerformanceCase",
            "--dataset-with-size-type",
            DatasetWithSizeType.LAIONLarge.value,
            "--filter-rate",
            "0.99",
            "--k",
            "1000000",
        ],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["task"].case_config
    assert case_config.k == 1_000_000
    assert case_config.case.dataset.data.name == "LAION"
    assert case_config.case.filters.filter_rate == 0.99


def test_cli_applies_vector_payload_to_standard_performance_case(monkeypatch: MonkeyPatch):
    result, captured = invoke_test_command(
        monkeypatch,
        [
            "--case-type",
            "Performance768D100M",
            "--payload-profile",
            "vector",
        ],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["task"].case_config
    assert case_config.payload_profile == PayloadProfile.VECTOR
    assert case_config.case.payload_profile == PayloadProfile.VECTOR


def test_cli_does_not_set_top_level_payload_for_capacity_case(monkeypatch: MonkeyPatch):
    result, captured = invoke_test_command(
        monkeypatch,
        ["--case-type", "CapacityDim128"],
    )

    assert result.exit_code == 0, result.output
    assert captured["task"].case_config.payload_profile is None
