import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients.milvus import cli as milvus_cli
from vectordb_bench.backend.clients.zilliz_cloud import cli as zilliz_cli
from vectordb_bench.cli import cli as common_cli


def test_milvus_cli_builds_shared_connection_config() -> None:
    parameters = {
        "db_label": "milvus-test",
        "uri": "http://localhost:19530",
        "user_name": "root",
        "password": "secret",
        "num_shards": "2",
        "replica_number": "3",
        "collection_name": "bench_collection",
    }

    config = milvus_cli._build_milvus_config(parameters)

    assert config.db_label == "milvus-test"
    assert config.uri.get_secret_value() == "http://localhost:19530"
    assert config.user == "root"
    assert config.password.get_secret_value() == "secret"
    assert config.num_shards == 2
    assert config.replica_number == 3
    assert config.collection_name == "bench_collection"

    parameters["password"] = None
    assert milvus_cli._build_milvus_config(parameters).password is None


def test_milvus_autoindex_cli_enables_partition_key_for_multitenant_case(
    monkeypatch: MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(milvus_cli, "run", fake_run)

    result = CliRunner().invoke(
        milvus_cli.MilvusAutoIndex,
        [
            "--case-type",
            "CloudMultiTenantSearchCase",
            "--uri",
            "http://localhost:19530",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["db_case_config"].use_partition_key is True


@pytest.mark.parametrize(
    ("level_args", "expected_level"),
    [(["--level", "2"], 2), ([], None)],
)
def test_milvus_autoindex_cli_handles_search_level(
    monkeypatch: MonkeyPatch,
    level_args: list[str],
    expected_level: int | None,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(milvus_cli, "run", fake_run)

    result = CliRunner().invoke(
        milvus_cli.MilvusAutoIndex,
        ["--uri", "http://localhost:19530", *level_args, "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    config = captured["db_case_config"]
    assert config.level == expected_level
    if expected_level is None:
        assert "params" not in config.search_param()


def test_milvus_flat_cli_rejects_search_level() -> None:
    result = CliRunner().invoke(
        milvus_cli.MilvusFlat,
        ["--uri", "http://localhost:19530", "--level", "2", "--dry-run"],
    )

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "--level" in result.output


@pytest.mark.parametrize("level", ["0", "11"])
def test_milvus_autoindex_cli_rejects_out_of_range_level(level: str) -> None:
    result = CliRunner().invoke(
        milvus_cli.MilvusAutoIndex,
        ["--uri", "http://localhost:19530", "--level", level, "--dry-run"],
    )

    assert result.exit_code == 2
    assert "1<=x<=10" in result.output


def test_zilliz_autoindex_cli_enables_partition_key_for_multitenant_case(
    monkeypatch: MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zilliz_cli, "run", fake_run)

    result = CliRunner().invoke(
        zilliz_cli.ZillizAutoIndex,
        [
            "--case-type",
            "CloudMultiTenantSearchCase",
            "--uri",
            "https://example.api.gcp-us-west1.zillizcloud.com",
            "--token",
            "secret",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["db_case_config"].use_partition_key is True


def test_milvus_autoindex_cli_nq(monkeypatch: MonkeyPatch) -> None:
    captured = {}

    def fake_run(tasks, task_label):
        captured["task"] = tasks[0]

    monkeypatch.setattr(common_cli.benchmark_runner, "run", fake_run)
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)

    result = CliRunner().invoke(
        milvus_cli.MilvusAutoIndex,
        ["--uri", "http://localhost:19530"],
    )
    assert result.exit_code == 0, result.output
    assert captured["task"].case_config.nq == 1

    result = CliRunner().invoke(
        milvus_cli.MilvusAutoIndex,
        ["--uri", "http://localhost:19530", "--nq", "2"],
    )
    assert result.exit_code == 0, result.output
    assert captured["task"].case_config.nq == 2

    result = CliRunner().invoke(
        milvus_cli.MilvusAutoIndex,
        ["--uri", "http://localhost:19530", "--nq", "0", "--dry-run"],
    )
    assert result.exit_code == 2, result.output
    assert "Invalid value for '--nq'" in result.output
