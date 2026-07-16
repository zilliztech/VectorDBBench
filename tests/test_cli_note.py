from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients.test import cli as test_cli
from vectordb_bench.cli import cli as common_cli


def invoke_test_command(monkeypatch: MonkeyPatch, args: list[str]):
    captured = {}

    def fake_run(tasks, task_label):
        captured["task"] = tasks[0]
        captured["task_label"] = task_label

    monkeypatch.setattr(common_cli.benchmark_runner, "run", fake_run)
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    result = CliRunner().invoke(test_cli.Test, args)
    return result, captured


def test_common_cli_exposes_note_options() -> None:
    result = CliRunner().invoke(test_cli.Test, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--note TEXT" in result.output
    assert "--note-file FILE" in result.output


def test_common_cli_stores_inline_note_in_task_config(monkeypatch: MonkeyPatch) -> None:
    note = '{"schema":"vdbbench-context/v1","deployment":"local"}'

    result, captured = invoke_test_command(monkeypatch, ["--note", note])

    assert result.exit_code == 0, result.output
    assert captured["task"].db_config.note == note


def test_common_cli_reads_note_file_into_task_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    note = '{\n  "schema": "vdbbench-context/v1",\n  "deployment": "managed"\n}'
    note_file = tmp_path / "run-context.json"
    note_file.write_text(note + "\n", encoding="utf-8")

    result, captured = invoke_test_command(monkeypatch, ["--note-file", str(note_file)])

    assert result.exit_code == 0, result.output
    assert captured["task"].db_config.note == note


def test_common_cli_rejects_inline_note_with_note_file(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    note_file = tmp_path / "run-context.json"
    note_file.write_text("context", encoding="utf-8")

    result, _ = invoke_test_command(
        monkeypatch,
        ["--note", "inline", "--note-file", str(note_file)],
    )

    assert result.exit_code != 0
    assert "--note and --note-file cannot be used together" in result.output


def test_common_cli_rejects_empty_note_file(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    note_file = tmp_path / "empty.txt"
    note_file.write_text("", encoding="utf-8")

    result, _ = invoke_test_command(monkeypatch, ["--note-file", str(note_file)])

    assert result.exit_code != 0
    assert "Note file is empty" in result.output


def test_common_cli_rejects_non_utf8_note_file(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    note_file = tmp_path / "invalid.txt"
    with note_file.open("wb") as output:
        output.write(b"\xff")

    result, _ = invoke_test_command(monkeypatch, ["--note-file", str(note_file)])

    assert result.exit_code != 0
    assert "Note file is not valid UTF-8" in result.output
