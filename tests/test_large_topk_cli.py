from click.testing import CliRunner
from pydantic import ValidationError
import pytest

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.test import cli as test_cli
from vectordb_bench.models import CaseConfig


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
