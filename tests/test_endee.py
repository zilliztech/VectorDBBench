import importlib
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.endee import cli as endee_cli
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.cli import cli as common_cli


def invoke_endee_cli(monkeypatch: pytest.MonkeyPatch, args: list[str]):
    captured = {}

    def fake_run(tasks, task_label):
        captured["task"] = tasks[0]
        captured["task_label"] = task_label

    monkeypatch.setattr(common_cli.benchmark_runner, "run", fake_run)
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    result = CliRunner().invoke(endee_cli.Endee, ["--token", "secret", *args])
    return result, captured


def test_endee_cli_builds_task_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    result, captured = invoke_endee_cli(monkeypatch, ["--collection-name", "bench"])

    assert result.exit_code == 0, result.output
    task = captured["task"]
    assert task.db == DB.Endee
    db_config = task.db_config.to_dict()
    assert db_config["token"] == "secret"
    assert db_config["collection_name"] == "bench"
    assert db_config["base_url"] is None
    assert db_config["precision"] == "int16"
    assert (db_config["m"], db_config["ef_con"], db_config["ef_search"]) == (16, 128, 128)
    assert db_config["prefilter_cardinality_threshold"] == 10_000
    assert db_config["filter_boost_percentage"] == 0
    assert task.db_config.version == ""


def test_endee_cli_passes_custom_options(monkeypatch: pytest.MonkeyPatch) -> None:
    result, captured = invoke_endee_cli(
        monkeypatch,
        [
            "--collection-name",
            "bench",
            "--base-url",
            "http://localhost:8080/api/v2",
            "--precision",
            "int8",
            "--version",
            "2.2.0",
            "--prefilter-cardinality-threshold",
            "5000",
            "--filter-boost-percentage",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    db_config = captured["task"].db_config
    assert db_config.version == "2.2.0"
    assert db_config.to_dict()["base_url"] == "http://localhost:8080/api/v2"
    assert db_config.to_dict()["precision"] == "int8"
    assert db_config.to_dict()["prefilter_cardinality_threshold"] == 5000
    assert db_config.to_dict()["filter_boost_percentage"] == 20


def test_endee_cli_accepts_deprecated_index_name_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    result, captured = invoke_endee_cli(monkeypatch, ["--index-name", "legacy"])

    assert result.exit_code == 0, result.output
    assert captured["task"].db_config.to_dict()["collection_name"] == "legacy"


def test_endee_cli_requires_collection_name(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = invoke_endee_cli(monkeypatch, [])

    assert result.exit_code != 0
    assert "--collection-name" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--prefilter-cardinality-threshold", "999"],
        ["--prefilter-cardinality-threshold", "1000001"],
        ["--filter-boost-percentage", "101"],
        ["--ef-search", "1025"],
    ],
)
def test_endee_cli_rejects_out_of_range_filter_tuning(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    result, _ = invoke_endee_cli(monkeypatch, ["--collection-name", "bench", *args])

    assert result.exit_code != 0


def test_endee_cli_reads_note_file_into_task_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    note = '{"schema":"vdbbench-context/v1","deployment":"endee-local"}'
    note_file = tmp_path / "endee-run-context.json"
    note_file.write_text(note, encoding="utf-8")

    result, captured = invoke_endee_cli(monkeypatch, ["--collection-name", "bench", "--note-file", str(note_file)])

    assert result.exit_code == 0, result.output
    assert captured["task"].db_config.note == note


@pytest.fixture
def endee_module(monkeypatch: pytest.MonkeyPatch):
    """Import the Endee client with a mocked ``endee`` SDK."""
    fake_sdk = MagicMock()
    fake_pkg = types.ModuleType("endee")
    fake_pkg.endee = fake_sdk
    monkeypatch.setitem(sys.modules, "endee", fake_pkg)
    monkeypatch.setitem(sys.modules, "endee.endee", fake_sdk)
    monkeypatch.delitem(sys.modules, "vectordb_bench.backend.clients.endee.endee", raising=False)

    endee_client = importlib.import_module("vectordb_bench.backend.clients.endee.endee")
    return endee_client, fake_sdk


def make_client(endee_module, **config):
    endee_client, fake_sdk = endee_module
    nd = fake_sdk.Endee.return_value
    nd.list_collections.return_value = [{"name": "bench"}]
    collection = nd.get_collection.return_value
    collection.search.return_value = {"results": {"dense": [{"id": "1"}, {"id": "2"}]}}

    db_config = {
        "token": "secret",
        "collection_name": "bench",
        "ef_search": 128,
        "prefilter_cardinality_threshold": 5000,
        "filter_boost_percentage": 20,
        **config,
    }
    client = endee_client.Endee(dim=4, db_config=db_config, db_case_config=None)
    return client, collection


def test_search_without_filter_omits_filter_tuning(endee_module) -> None:
    client, collection = make_client(endee_module)
    client.prepare_filter(MagicMock(type=FilterOp.NonFilter))

    assert client.search_embedding([0.1, 0.2, 0.3, 0.4], k=2) == [1, 2]

    kwargs = collection.search.call_args.kwargs
    assert kwargs["filter"] is None
    assert kwargs["ef_search"] == 128
    assert "prefilter_cardinality_threshold" not in kwargs
    assert "filter_boost_percentage" not in kwargs


def test_filtered_search_passes_filter_tuning(endee_module) -> None:
    client, collection = make_client(endee_module)
    client.prepare_filter(MagicMock(type=FilterOp.NumGE, int_value=10))

    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=2)

    kwargs = collection.search.call_args.kwargs
    assert kwargs["filter"] == [{"id": {"$gte": 10}}]
    assert kwargs["prefilter_cardinality_threshold"] == 5000
    assert kwargs["filter_boost_percentage"] == 20


def test_insert_builds_collection_objects(endee_module) -> None:
    client, collection = make_client(endee_module)
    client.with_scalar_labels = True

    count, err = client.insert_embeddings([[0.1] * 4, [0.2] * 4], [7, 8], labels_data=["a", "b"])

    assert (count, err) == (2, None)
    objects = collection.upsert.call_args.args[0]
    assert objects[0] == {
        "id": "7",
        "meta": {"id": 7},
        "filter": {"id": 7, "label": "a"},
        "fields": {"dense": [0.1] * 4},
    }


def test_insert_splits_batches_over_sdk_limit(endee_module) -> None:
    endee_client, _ = endee_module
    client, collection = make_client(endee_module)
    total = endee_client._MAX_UPSERT_BATCH + 5

    count, err = client.insert_embeddings([[0.0] * 4] * total, list(range(total)))

    assert (count, err) == (total, None)
    sizes = [len(call.args[0]) for call in collection.upsert.call_args_list]
    assert sizes == [endee_client._MAX_UPSERT_BATCH, 5]
