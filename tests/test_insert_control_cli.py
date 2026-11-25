import pytest
from click.testing import CliRunner

from vectordb_bench.backend.clients.memorydb import cli as memorydb_cli
from vectordb_bench.backend.clients.memorydb.config import MemoryDBHNSWConfig
from vectordb_bench.backend.clients.test import cli as test_cli
from vectordb_bench.cli import cli as core_cli


@pytest.mark.parametrize(
    ("args", "expected_batch_size"),
    [
        (["--dry-run"], 100),
        (["--dry-run", "--insert-batch-size", "250"], 250),
    ],
)
def test_common_insert_batch_size_is_forwarded_to_task_config(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expected_batch_size: int,
) -> None:
    captured = {}

    class FakeTaskConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(core_cli, "TaskConfig", FakeTaskConfig)

    result = CliRunner().invoke(test_cli.Test, args)

    assert result.exit_code == 0, result.output
    assert captured["insert_batch_size"] == expected_batch_size


@pytest.mark.parametrize("option", ["--insert-batch-size", "--streaming-insert-rate"])
def test_positive_insert_controls_reject_zero(option: str) -> None:
    result = CliRunner().invoke(test_cli.Test, ["--dry-run", option, "0"])

    assert result.exit_code == 2
    assert "x>=1" in result.output


@pytest.mark.parametrize(
    "case_type",
    ["StreamingPerformanceCase", "StreamingCustomDataset"],
)
def test_streaming_insert_rate_only_maps_to_streaming_cases(case_type: str) -> None:
    streaming = core_cli.get_custom_case_config(
        {
            "case_type": case_type,
            "dataset_with_size_type": None,
            "streaming_insert_rate": 750,
        },
    )
    non_streaming = core_cli.get_custom_case_config(
        {
            "case_type": "Performance1536D50K",
            "dataset_with_size_type": None,
            "streaming_insert_rate": 750,
        },
    )

    assert streaming == {"insert_rate": 750}
    assert non_streaming == {}


def test_cloud_insert_no_longer_has_a_custom_batch_mapping() -> None:
    custom_case = core_cli.get_custom_case_config(
        {
            "case_type": "CloudInsertCase",
            "dataset_with_size_type": None,
            "cloud_insert_duration": None,
            "cloud_insert_readiness_timeout": None,
            "cloud_insert_readiness_poll_interval": None,
        },
    )

    assert "batch_size" not in custom_case


def test_memorydb_cli_keeps_task_and_pipeline_batch_sizes_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(memorydb_cli, "run", fake_run)

    result = CliRunner().invoke(
        memorydb_cli.MemoryDB,
        [
            "--host",
            "localhost",
            "--dry-run",
            "--insert-batch-size",
            "200",
            "--memorydb-pipeline-batch-size",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["insert_batch_size"] == 200
    assert captured["db_case_config"].pipeline_batch_size == 8


def test_memorydb_config_accepts_legacy_insert_batch_size() -> None:
    config = MemoryDBHNSWConfig.model_validate({"insert_batch_size": 12})

    assert config.pipeline_batch_size == 12
    assert config.model_dump()["pipeline_batch_size"] == 12
    assert "insert_batch_size" not in config.model_dump()


def test_memorydb_config_prefers_canonical_pipeline_batch_size() -> None:
    config = MemoryDBHNSWConfig.model_validate(
        {
            "pipeline_batch_size": 8,
            "insert_batch_size": 12,
        },
    )

    assert config.pipeline_batch_size == 8


def test_removed_cloud_insert_batch_option_is_rejected() -> None:
    result = CliRunner().invoke(test_cli.Test, ["--dry-run", "--cloud-insert-batch-size", "5000"])

    assert result.exit_code == 2
    assert "No such option '--cloud-insert-batch-size'" in result.output
