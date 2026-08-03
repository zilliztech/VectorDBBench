from typing import Any

import pytest
from click import BadParameter

from vectordb_bench.backend.clients.adbpg.adbpg import Adbpg
from vectordb_bench.backend.clients.adbpg.config import AdbpgIndexConfig
from vectordb_bench.backend.clients.adbpg.options import parse_key_values, parse_reloptions
from vectordb_bench.backend.clients.api import MetricType


def _config(**kwargs: Any) -> AdbpgIndexConfig:
    return AdbpgIndexConfig(metric_type=MetricType.COSINE, **kwargs)


def test_generic_session_gucs_override_builtin_values() -> None:
    config = _config(
        session_gucs={
            "fastann.hnsw_ef_search": "321",
            "fastann.new_search_switch": "off",
        }
    )

    options = {
        item["parameter"]["setting_name"]: item["parameter"]["val"]
        for item in config.session_param()["session_options"]
    }
    assert options["fastann.hnsw_ef_search"] == "321"
    assert options["fastann.new_search_switch"] == "off"


def test_cli_parses_generic_settings_and_reset() -> None:
    assert parse_key_values(None, None, ("a=1,b=2", "a=3")) == {"a": "3", "b": "2"}
    assert parse_reloptions(None, None, ("nova_ef_search=120", "nova_nprobe")) == {
        "nova_ef_search": "120",
        "nova_nprobe": None,
    }
    with pytest.raises(BadParameter):
        parse_key_values(None, None, ("missing_value",))


def test_search_setup_uses_coordinator_and_resolves_placeholders() -> None:
    executed: list[str] = []
    connected_with: dict[str, Any] = {}

    class FakeCursor:
        def execute(self, query: Any) -> None:
            executed.append(query if isinstance(query, str) else query.as_string(None))

        def close(self) -> None:
            pass

    class FakeConnection:
        commits = 0
        rollbacks = 0

        def commit(self) -> None:
            self.commits += 1

        def rollback(self) -> None:
            self.rollbacks += 1

        def close(self) -> None:
            pass

    client = object.__new__(Adbpg)
    client.name = "Adbpg"
    client.table_name = "docs"
    client._index_name = "docs_novam_index"
    client.connect_config = {"host": "db", "options": "-c gp_session_role=utility"}
    client.case_config = _config(
        benchmark_topk=10,
        index_reloptions={
            "nova_autotune_topk": "$topk",
            "nova_autotune_recall": "0.95",
            "nova_ef_search": None,
        },
        setup_sql=("SELECT $topk FROM $table",),
    )
    connection = FakeConnection()
    cursor = FakeCursor()
    client._create_connection = lambda **kwargs: (connected_with.update(kwargs) or (connection, cursor))

    client._apply_search_setup()

    assert connected_with == {"host": "db"}
    assert executed == [
        'ALTER INDEX "public"."docs_novam_index" SET '
        '("nova_autotune_topk" = 10, "nova_autotune_recall" = \'0.95\')',
        'ALTER INDEX "public"."docs_novam_index" RESET ("nova_ef_search")',
        'SELECT 10 FROM "public"."docs"',
    ]
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert client._render_setup_sql("SELECT $1; DO $body$ BEGIN NULL; END $body$") == (
        "SELECT $1; DO $body$ BEGIN NULL; END $body$"
    )


def test_optimize_applies_setup_after_index_work() -> None:
    events: list[str] = []
    client = object.__new__(Adbpg)
    client._post_insert = lambda: events.append("index")
    client._apply_search_setup = lambda: events.append("setup")

    client.optimize()

    assert events == ["index", "setup"]
