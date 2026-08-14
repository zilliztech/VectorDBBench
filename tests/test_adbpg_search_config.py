from pathlib import Path
from typing import Any

import pytest
from click import BadParameter
from click.testing import CliRunner
from psycopg import adapters as psycopg_adapters

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.adbpg import cli as adbpg_cli
from vectordb_bench.backend.clients.adbpg.adbpg import Adbpg
from vectordb_bench.backend.clients.adbpg.config import AdbpgConfig, AdbpgIndexConfig
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


def test_index_build_reloptions_override_dedicated_defaults() -> None:
    config = _config(
        algorithm="novamr",
        index_build_reloptions={
            "algorithm": "novadflat",
            "hnsw_m": "64",
            "future_build_option": "enabled",
        },
    )

    options = config.index_param()["index_creation_with_options"]
    by_name = {option["option_name"]: option["val"] for option in options}
    assert by_name["algorithm"] == "novadflat"
    assert by_name["hnsw_m"] == "64"
    assert by_name["future_build_option"] == "enabled"
    assert sum(option["option_name"] == "algorithm" for option in options) == 1

    with pytest.raises(ValueError, match="cannot override dataset-derived options: dim"):
        _config(index_build_reloptions={"dim": "384"}).index_param()


def test_index_build_reloptions_render_in_create_index(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr("vectordb_bench.backend.clients.adbpg.adbpg.log.debug", lambda *_args: None)

    class FakeCursor:
        connection = None
        adapters = psycopg_adapters

        def execute(self, query: Any) -> "FakeCursor":
            executed.append(query if isinstance(query, str) else query.as_string(None))
            return self

        def fetchall(self) -> list[Any]:
            return []

    class FakeConnection:
        def commit(self) -> None:
            pass

    client = object.__new__(Adbpg)
    client.name = "Adbpg"
    client.dim = 768
    client.table_name = "vector"
    client._index_name = "vector_novamr_index"
    client._vector_field = "embedding"
    client._primary_field = "id"
    client.conn = FakeConnection()
    client.cursor = FakeCursor()
    client.case_config = _config(
        index_build_includes=("id", "label", "tenant_id"),
        index_build_reloptions={
            "algorithm": "novamr",
            "hnsw_m": "48",
            "hnsw_ef_construction": "600",
            "rabitq_bits": "7",
            "auto_reduction": "on",
        },
    )

    client._create_index()

    create_sql = next(statement for statement in executed if "CREATE INDEX" in statement)
    assert "\"algorithm\" = 'novamr'" in create_sql
    assert "\"hnsw_m\" = '48'" in create_sql
    assert "\"hnsw_ef_construction\" = '600'" in create_sql
    assert "\"rabitq_bits\" = '7'" in create_sql
    assert "\"auto_reduction\" = 'on'" in create_sql
    assert create_sql.count('"algorithm"') == 1
    assert 'INCLUDE ("id", "label", "tenant_id")' in create_sql


def test_index_build_include_defaults_and_validation() -> None:
    assert _config().index_build_includes == ("id",)
    assert _config(index_build_includes=("id", "label", "id")).index_build_includes == ("id", "label")

    with pytest.raises(ValueError, match="column cannot be empty"):
        _config(index_build_includes=("id", " "))


def test_cli_parses_generic_settings_and_reset() -> None:
    assert parse_key_values(None, None, ("a=1", "b=2", "a=3")) == {"a": "3", "b": "2"}
    assert parse_key_values(None, None, ("search_path=foo,bar",)) == {"search_path": "foo,bar"}
    assert parse_reloptions(None, None, ("nova_ef_search=120", "nova_nprobe")) == {
        "nova_ef_search": "120",
        "nova_nprobe": None,
    }
    with pytest.raises(BadParameter):
        parse_key_values(None, None, ("missing_value",))


def test_cohere_autotune_example_loads_through_click(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(*, db: DB, db_config: AdbpgConfig, db_case_config: AdbpgIndexConfig, **parameters: Any) -> None:
        captured.update(
            db=db,
            db_config=db_config,
            db_case_config=db_case_config,
            parameters=parameters,
        )

    monkeypatch.setattr(adbpg_cli, "run", fake_run)
    config_file = Path(__file__).parents[1] / "vectordb_bench" / "config-files" / "adbpg_cohere1m_autotune.yml"

    result = CliRunner().invoke(
        adbpg_cli.AdbpgNova,
        ["--config-file", str(config_file), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert captured["db"] == DB.Adbpg
    assert captured["db_config"].db_label == "cohere1m-novamr-autotune"
    assert captured["parameters"]["case_type"] == "Performance768D1M"
    assert captured["parameters"]["k"] == 10
    assert captured["parameters"]["drop_old"] is True
    assert captured["parameters"]["load"] is True
    case_config = captured["db_case_config"]
    assert case_config.index_build_reloptions == {
        "algorithm": "novamr",
        "hnsw_m": "48",
        "hnsw_ef_construction": "600",
        "rabitq_bits": "7",
        "auto_reduction": "on",
    }
    assert case_config.index_build_includes == ("id",)
    assert case_config.session_gucs == {"fastann.nova_adaptive_gamma": "0"}
    assert case_config.index_reset_reloptions == {}
    assert case_config.autotune_params == {
        "topk": "10",
        "target_recall": "0.95",
        "n_samples": "300",
        "n_trials": "500",
        "n_threads": "32",
    }
    assert case_config.autotune_timeout == 600
    assert case_config.setup_sql == ('ANALYZE "public"."vector"',)
    assert "$" not in config_file.read_text()


def test_search_setup_uses_coordinator_and_preserves_sql() -> None:
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
        index_reset_reloptions={
            "nova_autotune_topk": "10",
            "nova_autotune_recall": "0.95",
            "nova_ef_search": None,
        },
        setup_sql=(
            'ANALYZE "public"."docs"',
            "DO $$ BEGIN NULL; END $$;",
            "SELECT '$topk', '$table', '$index'",
        ),
    )
    connection = FakeConnection()
    cursor = FakeCursor()
    client._create_connection = lambda **kwargs: (connected_with.update(kwargs) or (connection, cursor))

    client._apply_search_setup()

    assert connected_with == {"host": "db"}
    expected_set_sql = "".join(
        (
            'ALTER INDEX "public"."docs_novam_index" SET ',
            "(\"nova_autotune_topk\" = '10', \"nova_autotune_recall\" = '0.95')",
        )
    )
    assert executed == [
        'ANALYZE "public"."docs"',
        "DO $$ BEGIN NULL; END $$;",
        "SELECT '$topk', '$table', '$index'",
        expected_set_sql,
        'ALTER INDEX "public"."docs_novam_index" RESET ("nova_ef_search")',
    ]
    assert connection.commits == 2
    assert connection.rollbacks == 0


def test_autotune_waits_for_all_config_rows_and_worker_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[tuple[str, tuple[Any, ...] | None]] = []

    class FakeCursor:
        current_query = ""
        progress_polls = 0

        def execute(self, query: Any, params: tuple[Any, ...] | None = None) -> "FakeCursor":
            self.current_query = query if isinstance(query, str) else query.as_string(None)
            executed.append((self.current_query, params))
            return self

        def fetchone(self) -> tuple[Any, ...] | None:
            if "SELECT fastann.nova_autotune(" in self.current_query:
                return (12345,)
            if "nova_autotune_progress" in self.current_query:
                self.progress_polls += 1
                if self.progress_polls == 1:
                    return (4321, "running", 10, 20, 4)
                return None
            if "nova_autotune_configs" in self.current_query:
                return (4,)
            raise AssertionError(self.current_query)

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
    client.table_name = "vector"
    client._index_name = "vector_novamr_index"
    client.connect_config = {"host": "db", "options": "-c gp_session_role=utility"}
    client.case_config = _config(
        setup_sql=('ANALYZE "public"."vector"',),
        autotune_params={
            "topk": "ARRAY[10,100]",
            "target_recall": "ARRAY[0.90,0.95]",
            "n_samples": "300",
            "synthetic_query_mode": "'anchored_gaussian'",
        },
    )
    connection = FakeConnection()
    cursor = FakeCursor()
    client._create_connection = lambda **_kwargs: (connection, cursor)
    monkeypatch.setattr("vectordb_bench.backend.clients.adbpg.adbpg.time.sleep", lambda _seconds: None)

    client._apply_search_setup()

    sql_text = [query for query, _params in executed]
    assert sql_text[0] == 'ANALYZE "public"."vector"'
    assert "SELECT fastann.nova_autotune(" in sql_text[1]
    assert '"topk" => ARRAY[10,100]' in sql_text[1]
    assert '"target_recall" => ARRAY[0.90,0.95]' in sql_text[1]
    assert '"n_samples" => 300' in sql_text[1]
    assert "\"synthetic_query_mode\" => 'anchored_gaussian'" in sql_text[1]
    assert executed[1][1] == ("public.vector_novamr_index",)
    assert "nova_autotune_progress" in sql_text[2]
    assert "nova_autotune_configs" in sql_text[3]
    assert "c.topk = ANY(%s::integer[])" in sql_text[3]
    assert "c.target_recall = ANY(%s::real[])" in sql_text[3]
    assert executed[3][1] == (
        "public.vector_novamr_index",
        [10, 100],
        [0.90, 0.95],
    )
    assert sum("nova_autotune_progress" in query for query in sql_text) == 2
    assert sum("nova_autotune_configs" in query for query in sql_text) == 2
    assert all("nova_autotune_status" not in query for query in sql_text)
    assert connection.commits == 4
    assert connection.rollbacks == 0


def test_autotune_and_target_reloptions_are_separate_modes() -> None:
    client = object.__new__(Adbpg)
    client.case_config = _config(
        autotune_params={"topk": "10", "target_recall": "0.95"},
        index_reset_reloptions={"nova_autotune_topk": "10", "nova_autotune_recall": "0.95"},
    )

    with pytest.raises(ValueError, match="Do not combine autotune_param"):
        client._validate_autotune_configuration(drop_old=True)


def test_autotune_fails_closed_when_config_row_count_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCursor:
        current_query = ""

        def execute(self, query: Any, _params: tuple[Any, ...] | None = None) -> "FakeCursor":
            self.current_query = query if isinstance(query, str) else query.as_string(None)
            return self

        def fetchone(self) -> tuple[Any, ...] | None:
            if "SELECT fastann.nova_autotune(" in self.current_query:
                return (12345,)
            if "nova_autotune_progress" in self.current_query:
                return None
            if "nova_autotune_configs" in self.current_query:
                return (0,)
            raise AssertionError(self.current_query)

    class FakeConnection:
        def commit(self) -> None:
            pass

    client = object.__new__(Adbpg)
    client._index_name = "vector_novamr_index"
    client.case_config = _config(autotune_params={"topk": "10", "target_recall": "0.95"})
    monkeypatch.setattr("vectordb_bench.backend.clients.adbpg.adbpg.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match=r"finished with 0 config rows.*expected 1"):
        client._run_nova_autotune(FakeConnection(), FakeCursor())


def test_autotune_requires_new_index_and_explicit_topk() -> None:
    client = object.__new__(Adbpg)
    client.case_config = _config(autotune_params={"topk": "ARRAY[10,100]", "target_recall": "ARRAY[0.90,0.95]"})

    with pytest.raises(ValueError, match="requires a load run"):
        client._validate_autotune_configuration(drop_old=False)

    client.case_config = _config(autotune_params={"target_recall": "0.95"})
    with pytest.raises(ValueError, match="requires an explicit topk"):
        client._validate_autotune_configuration(drop_old=True)

    client.case_config = _config(autotune_params={"topk": "ARRAY[10,100]", "target_recall": "ARRAY[0.90,0.95]"})
    assert client._autotune_topks() == (10, 100)
    assert client._autotune_target_recalls() == (0.90, 0.95)

    client.case_config = _config(autotune_params={"topk": "ARRAY[10,10]", "target_recall": "ARRAY[0.95,0.95]"})
    assert client._autotune_topks() == (10,)
    assert client._autotune_target_recalls() == (0.95,)


def test_optimize_applies_setup_after_index_work() -> None:
    events: list[str] = []
    client = object.__new__(Adbpg)
    client._post_insert = lambda: events.append("index")
    client._apply_search_setup = lambda: events.append("setup")

    client.optimize()

    assert events == ["index", "setup"]
