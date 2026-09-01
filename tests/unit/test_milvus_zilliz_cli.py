from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients.milvus import cli as milvus_cli
from vectordb_bench.backend.clients.zilliz_cloud import cli as zilliz_cli


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
