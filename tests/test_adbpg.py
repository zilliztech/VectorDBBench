from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from click.testing import CliRunner
from pydantic import SecretStr, ValidationError

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.adbpg import cli as adbpg_cli
from vectordb_bench.backend.clients.adbpg.adbpg import Adbpg, AdbpgTimeoutError
from vectordb_bench.backend.clients.adbpg.config import (
    AdbpgAutotuneParameters,
    AdbpgConfig,
    AdbpgIndexConfig,
    AdbpgParameterGroup,
)
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.cli.batch_cli import build_sub_cmd_args
from vectordb_bench.models import CaseConfig, CaseType, TaskConfig, TaskStage, TestResult

if TYPE_CHECKING:
    from pathlib import Path


def make_index_config(**overrides: Any) -> AdbpgIndexConfig:
    base = {
        "metric_type": MetricType.COSINE,
        "build_parameters": {
            "reloption": {
                "algorithm": "novamr",
                "note": "quotes ', comma, and $$ stay exact",
            },
            "guc": {},
        },
        "search_parameters": {
            "reloption": {
                "nova_autotune_topk": 10,
                "nova_autotune_recall": 0.95,
            },
            "guc": {"search_path": "foo,bar"},
        },
        "autotune_parameters": {
            "topk": [10, 100],
            "target_recall": [0.9, 0.95],
            "n_threads": 4,
            "future_optimizer": {"levels": [1, 2]},
        },
    }
    base.update(overrides)
    return AdbpgIndexConfig(**base)


class FakeCursor:
    def __init__(
        self,
        fail_on: str | None = None,
        error: Exception | None = None,
        fetchone_result: tuple[Any, ...] = (1, True, 2, True, True),
    ):
        self.executions: list[tuple[Any, Any]] = []
        self.fail_on = fail_on
        self.error = error or RuntimeError("database error")
        self.fetchone_result = fetchone_result
        self.connection: FakeConnection | None = None
        self.closed = False

    def execute(self, query: Any, params: Any = None, **kwargs: Any):
        self.executions.append((query, params))
        if self.connection is not None and self.connection.aborted:
            raise psycopg.errors.InFailedSqlTransaction
        if self.fail_on and self.fail_on in str(query):
            if self.connection is not None:
                self.connection.aborted = True
            raise self.error
        return self

    def fetchall(self) -> list[tuple[str]]:
        return [("ok",)]

    def fetchone(self) -> tuple[Any, ...]:
        return self.fetchone_result

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.aborted = False
        self.closed = False
        cursor.connection = self

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1
        self.aborted = False

    def close(self) -> None:
        self.closed = True


def make_client(*, with_scalar_labels: bool = False) -> Adbpg:
    client = Adbpg.__new__(Adbpg)
    client.name = "Adbpg"
    client.case_config = make_index_config()
    client.table_name = "vector$table"
    client.connect_config = {}
    client.dim = 768
    client.with_scalar_labels = with_scalar_labels
    client._primary_field = "id"
    client._vector_field = "embedding"
    client._scalar_label_field = "label"
    client._index_name = "vector$table_novamr_index"
    client.where_clause = ""
    return client


class TestAdbpgConfig:
    def test_defaults_allow_construction_without_password(self):
        cfg = AdbpgConfig(db_label="", version="", note="")
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.db_name == "postgres"
        assert cfg.password.get_secret_value() == ""

    def test_to_dict_uses_normal_coordinator_session(self):
        cfg = AdbpgConfig(
            user_name=SecretStr("u"),
            password=SecretStr("pw"),
            host="h.example.com",
            port=5432,
            db_name="postgres",
        )
        connect_config = cfg.to_dict()["connect_config"]
        assert connect_config == {
            "host": "h.example.com",
            "port": 5432,
            "dbname": "postgres",
            "user": "u",
            "password": "pw",
        }


class TestAdbpgStructuredConfig:
    def test_free_form_names_and_scalar_types_are_preserved(self):
        group = AdbpgParameterGroup(
            reloption={"future.knob": 7, "enabled": True, "reset_me": None},
            guc={"search_path": "foo,bar", "ratio": 0.5},
        )
        assert group.model_dump() == {
            "reloption": {"future.knob": 7, "enabled": True, "reset_me": None},
            "guc": {"search_path": "foo,bar", "ratio": 0.5},
        }

    @pytest.mark.parametrize(
        "value",
        [
            {"reloption": {"bad": [1, 2]}},
            {"guc": {"bad": {"nested": True}}},
            {"unsupported": {}},
        ],
    )
    def test_invalid_parameter_structures_are_rejected(self, value: dict[str, Any]):
        with pytest.raises(ValidationError):
            AdbpgParameterGroup(**value)

    def test_build_defaults_are_merged_with_user_parameters(self):
        build = AdbpgIndexConfig(
            build_parameters={"reloption": {"algorithm": "novamr"}},
        ).build_parameters
        assert build.model_dump() == {
            "reloption": {
                "algorithm": "novamr",
                "hnsw_m": 48,
                "hnsw_ef_construction": 600,
                "rabitq_bits": 7,
                "auto_reduction": False,
            },
            "guc": {"fastann.build_parallel_processes": 32},
        }

    def test_default_algorithm_is_used_for_every_entry_point(self):
        assert AdbpgIndexConfig().algorithm == "novamr"

    def test_autotune_defaults_and_preserves_multiple_targets(self):
        autotune = AdbpgAutotuneParameters(
            topk=[10, 100],
            target_recall=[0.9, 0.95],
        )
        assert autotune.topk == [10, 100]
        assert autotune.target_recall == [0.9, 0.95]
        assert autotune.n_samples == 200
        assert autotune.n_trials == 200
        assert autotune.timeout == 600

    def test_autotune_preserves_extra_json_parameters_and_allows_zero_samples(self):
        autotune = AdbpgAutotuneParameters(
            topk=[10],
            target_recall=[0.95],
            n_samples=0,
            n_trials=100,
            timeout=600,
            n_threads=4,
            future_optimizer={"levels": [1, 2]},
            future_optional=None,
        )
        assert autotune.model_dump(mode="json")["n_threads"] == 4
        assert autotune.model_dump(mode="json")["future_optimizer"] == {"levels": [1, 2]}
        assert "future_optional" in autotune.model_dump(mode="json")
        assert autotune.model_dump(mode="json")["future_optional"] is None

    def test_autotune_is_disabled_when_configuration_is_omitted(self):
        assert AdbpgIndexConfig().autotune_parameters is None
        assert AdbpgIndexConfig(autotune_parameters="").autotune_parameters is None

    def test_legacy_enable_switch_is_rejected(self):
        with pytest.raises(ValidationError, match="omit autotune_parameters"):
            AdbpgAutotuneParameters(
                enable=False,
                topk=[10],
                target_recall=[0.95],
            )

    @pytest.mark.parametrize(
        "value",
        [
            {},
            {"topk": [0], "target_recall": [0.95], "n_samples": 1, "n_trials": 1, "timeout": 1},
            {"topk": [10], "target_recall": [1.1], "n_samples": 1, "n_trials": 1, "timeout": 1},
            {"topk": [10], "target_recall": [0.95], "n_samples": -1, "n_trials": 1, "timeout": 1},
        ],
    )
    def test_invalid_autotune_is_rejected(self, value: dict[str, Any]):
        with pytest.raises(ValidationError):
            AdbpgAutotuneParameters(**value)

    def test_old_flattened_fields_are_migrated_to_structured_parameters(self):
        config = AdbpgIndexConfig(
            algorithm="novad",
            hnsw_m=32,
            ef_construction=256,
            nlist=2048,
            rabitq_bits=6,
            auto_reduction=True,
            pca_dim=384,
            build_parallel_processes=16,
            ef_search=77,
            max_scan_points=900,
            quantize_rescore_amp=2.0,
            nova_adaptive_gamma=0.4,
            index_scan_mode="streaming",
            nprobe=9,
        )
        assert config.build_parameters.reloption == {
            "algorithm": "novad",
            "hnsw_m": 32,
            "hnsw_ef_construction": 256,
            "rabitq_bits": 6,
            "auto_reduction": True,
            "nlist": 2048,
            "pca_dim": 384,
        }
        assert config.build_parameters.guc == {"fastann.build_parallel_processes": 16}
        assert config.search_parameters.guc == {
            "fastann.hnsw_ef_search": 77,
            "fastann.hnsw_max_scan_points": 900,
            "fastann.quantize_rescore_amp": 2.0,
            "fastann.nova_adaptive_gamma": 0.4,
            "fastann.index_scan_mode": "streaming",
            "fastann.nova_nprobe": 9,
        }

    def test_structured_parameters_take_precedence_over_legacy_fields(self):
        config = AdbpgIndexConfig(
            algorithm="novad",
            ef_search=77,
            build_parameters={"reloption": {"algorithm": "novamr"}},
            search_parameters={"guc": {"fastann.hnsw_ef_search": 130}},
        )
        assert config.algorithm == "novamr"
        assert config.search_parameters.guc == {"fastann.hnsw_ef_search": 130}

    def test_unknown_fields_are_still_rejected(self):
        with pytest.raises(ValidationError, match="unknown"):
            AdbpgIndexConfig(unknown=1)


class TestAdbpgUdfPayload:
    def test_pure_vector_payload_is_complete_and_excludes_credentials(self):
        client = make_client()
        payload = client._build_udf_payload()
        assert payload["api_version"] == 1
        assert {"case_type", "workload_type"}.isdisjoint(payload)
        assert payload["relation"] == {
            "schema": "public",
            "table": "vector$table",
            "index": "vector$table_novamr_index",
        }
        assert payload["columns"] == [
            {"name": "id", "type": "bigint", "role": "primary_key"},
            {"name": "embedding", "type": "vector(768)", "role": "vector"},
        ]
        assert payload["metric"] == "cosine"
        assert set(payload["parameters"]) == {"build", "search", "autotune"}
        assert payload["parameters"]["autotune"]["topk"] == [10, 100]
        assert payload["parameters"]["autotune"]["n_samples"] == 200
        assert payload["parameters"]["autotune"]["n_trials"] == 200
        assert payload["parameters"]["autotune"]["timeout"] == 600
        assert payload["parameters"]["autotune"]["n_threads"] == 4
        assert payload["parameters"]["autotune"]["future_optimizer"] == {"levels": [1, 2]}
        assert payload["parameters"]["build"]["reloption"]["note"] == "quotes ', comma, and $$ stay exact"
        assert payload["parameters"]["search"] == {
            "reloption": {"nova_autotune_topk": 10, "nova_autotune_recall": 0.95},
            "guc": {"search_path": "foo,bar"},
        }
        assert {"password", "user", "user_name", "host", "port"}.isdisjoint(payload)

    def test_payload_omits_autotune_when_configuration_is_omitted(self):
        client = make_client()
        client.case_config = make_index_config(autotune_parameters=None)

        assert "autotune" not in client._build_udf_payload()["parameters"]

    def test_hybrid_payload_describes_filter_column(self):
        client = make_client(with_scalar_labels=True)
        payload = client._build_udf_payload()
        assert payload["columns"][-1] == {"name": "label", "type": "varchar(64)", "role": "filter"}


class TestAdbpgLifecycle:
    def test_optimize_calls_build_then_optimize_once_with_same_bound_payload(self):
        client = make_client()
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        client.optimize()

        udf_calls = [(query, params) for query, params in cursor.executions if "vectordbbench_" in str(query)]
        assert ["vectordbbench_build" in str(query) for query, _ in udf_calls] == [True, False]
        assert ["vectordbbench_optimize" in str(query) for query, _ in udf_calls] == [False, True]
        assert udf_calls[0][1][0].obj == udf_calls[1][1][0].obj
        statements = [str(query) for query, _ in cursor.executions]
        assert next(i for i, value in enumerate(statements) if "vectordbbench_optimize" in value) < next(
            i for i, value in enumerate(statements) if "ALTER INDEX" in value
        )
        assert sum("ALTER INDEX" in statement for statement in statements) == 1
        assert conn.commit_count == 3
        assert conn.rollback_count == 0
        assert any(params == ("630s",) for _, params in cursor.executions)

    def test_build_failure_rolls_back_and_skips_optimize(self):
        client = make_client()
        cursor = FakeCursor(fail_on="vectordbbench_build")
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        with pytest.raises(RuntimeError, match=r"fastann\.vectordbbench_build"):
            client.optimize()

        assert conn.rollback_count == 1
        assert not any("vectordbbench_optimize" in str(query) for query, _ in cursor.executions)

    def test_optimize_failure_rolls_back_and_drops_incomplete_index(self):
        client = make_client()
        cursor = FakeCursor(fail_on="vectordbbench_optimize")
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        with pytest.raises(RuntimeError, match=r"fastann\.vectordbbench_optimize"):
            client.optimize()

        assert conn.commit_count == 2
        assert conn.rollback_count == 2
        assert not any("ALTER INDEX" in str(query) for query, _ in cursor.executions)
        assert sum("DROP INDEX" in str(query) for query, _ in cursor.executions) == 1

    def test_optimize_statement_timeout_is_reported(self):
        client = make_client()
        cursor = FakeCursor(
            fail_on="vectordbbench_optimize",
            error=psycopg.errors.QueryCanceled("statement timeout"),
        )
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        with pytest.raises(AdbpgTimeoutError, match="630 seconds"):
            client.optimize()

        assert conn.rollback_count == 2
        assert sum("DROP INDEX" in str(query) for query, _ in cursor.executions) == 1

    def test_missing_udfs_fail_before_destructive_build_setup(self, monkeypatch: pytest.MonkeyPatch):
        cursor = FakeCursor(fetchone_result=(None, None, 2, True, True))
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))

        with pytest.raises(RuntimeError, match=r"vectordbbench_build\(jsonb\)"):
            Adbpg(
                dim=768,
                db_config={"table_name": "vector", "connect_config": {}},
                db_case_config=make_index_config(),
                drop_old=True,
            )

        assert len(cursor.executions) == 1
        assert "to_regprocedure" in str(cursor.executions[0][0])
        assert not any("DROP" in str(query) or "CREATE" in str(query) for query, _ in cursor.executions)
        assert cursor.closed
        assert conn.closed

    def test_missing_udf_privilege_fails_before_destructive_build_setup(self, monkeypatch: pytest.MonkeyPatch):
        cursor = FakeCursor(fetchone_result=(1, False, 2, True, True))
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))

        with pytest.raises(RuntimeError, match=r"vectordbbench_build\(jsonb\)"):
            Adbpg(
                dim=768,
                db_config={"table_name": "vector", "connect_config": {}},
                db_case_config=make_index_config(),
                drop_old=True,
            )

        assert len(cursor.executions) == 1
        assert cursor.closed
        assert conn.closed

    def test_reloption_failure_rolls_back_before_dropping_incomplete_index(self):
        client = make_client()
        cursor = FakeCursor(fail_on="ALTER INDEX")
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        with pytest.raises(RuntimeError, match="database error"):
            client.optimize()

        assert conn.rollback_count == 1
        assert sum("DROP INDEX" in str(query) for query, _ in cursor.executions) == 1
        assert conn.commit_count == 3

    def test_query_cancellation_without_configured_timeout_is_not_mislabeled(self):
        client = make_client()
        cursor = FakeCursor(
            fail_on="vectordbbench_build",
            error=psycopg.errors.QueryCanceled("canceled by user"),
        )
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        with pytest.raises(RuntimeError, match="failed: canceled by user"):
            client.optimize()

    def test_non_timeout_cancellation_with_configured_timeout_is_not_mislabeled(self):
        client = make_client()
        cursor = FakeCursor(
            fail_on="vectordbbench_optimize",
            error=psycopg.errors.QueryCanceled("canceling statement due to user request"),
        )
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        with pytest.raises(RuntimeError, match="failed: canceling statement due to user request"):
            client.optimize()

    def test_optimize_without_autotune_still_calls_both_udfs(self):
        client = make_client()
        client.case_config = make_index_config(autotune_parameters=None)
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        client.conn = conn
        client.cursor = cursor

        client.optimize()

        udf_calls = [(query, params) for query, params in cursor.executions if "vectordbbench_" in str(query)]
        assert ["vectordbbench_build" in str(query) for query, _ in udf_calls] == [True, False]
        assert ["vectordbbench_optimize" in str(query) for query, _ in udf_calls] == [False, True]
        assert "autotune" not in udf_calls[0][1][0].obj["parameters"]
        assert not any("statement_timeout" in str(query) for query, _ in cursor.executions)

    def test_reuse_constructor_applies_reloptions_once_and_calls_no_udf(self, monkeypatch: pytest.MonkeyPatch):
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))

        Adbpg(
            dim=768,
            db_config={"table_name": "vector", "connect_config": {}},
            db_case_config=make_index_config(
                search_parameters={
                    "reloption": {"nova_autotune_topk": 10, "nova_autotune_recall": None},
                    "guc": {},
                }
            ),
            drop_old=False,
        )

        statements = [str(query) for query, _ in cursor.executions]
        assert sum("ALTER INDEX" in statement for statement in statements) == 2
        assert not any("vectordbbench_" in statement for statement in statements)
        assert conn.commit_count == 1
        assert cursor.closed
        assert conn.closed

    def test_search_connections_only_apply_search_gucs(self, monkeypatch: pytest.MonkeyPatch):
        client = make_client()
        connections = []

        def create_connection(**_kwargs):
            cursor = FakeCursor()
            conn = FakeConnection(cursor)
            connections.append((conn, cursor))
            return conn, cursor

        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(create_connection))
        for _ in range(2):
            with client.init():
                client.prepare_filter(non_filter)

        assert len(connections) == 2
        for conn, cursor in connections:
            assert [(query, params) for query, params in cursor.executions] == [
                ("SELECT set_config(%s, %s, false)", ("search_path", "foo,bar"))
            ]
            assert conn.commit_count == 1
            assert not any(
                "ALTER INDEX" in str(query) or "vectordbbench_" in str(query) for query, _ in cursor.executions
            )

    def test_non_search_connections_do_not_apply_search_gucs(self, monkeypatch: pytest.MonkeyPatch):
        client = make_client()
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))

        with client.init():
            pass

        assert cursor.executions == []
        assert conn.commit_count == 0

    def test_udf_failure_prevents_search_runner_initialization(self, monkeypatch: pytest.MonkeyPatch):
        task = TaskConfig(
            db=DB.Adbpg,
            db_config=AdbpgConfig(),
            db_case_config=make_index_config(),
            case_config=CaseConfig(case_id=CaseType.Performance768D1M, k=10),
            stages=[TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_SERIAL],
        )
        runner = CaseRunner(
            run_id="run-id",
            config=task,
            ca=task.case_config.case,
            status=RunningStatus.PENDING,
            dataset_source=DatasetSource.S3,
        )
        search_started = []
        monkeypatch.setattr(CaseRunner, "_load_data", lambda _self: (1_000_000, 1.0))

        def fail_optimize(_self: CaseRunner) -> float:
            raise RuntimeError("fastann.vectordbbench_build failed")

        monkeypatch.setattr(CaseRunner, "_optimize", fail_optimize)
        monkeypatch.setattr(CaseRunner, "_init_search_runners", lambda _self: search_started.append(True))

        with pytest.raises(RuntimeError, match=r"vectordbbench_build failed"):
            runner._run_perf_case(drop_old=True)
        assert search_started == []


def test_single_and_batch_cli_preserve_equivalent_structured_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = []
    monkeypatch.setattr(
        adbpg_cli,
        "run",
        lambda **kwargs: captured.append((kwargs["db_config"], kwargs["db_case_config"])),
    )
    values = {
        "build_parameters": {"reloption": {"algorithm": "novamr", "hnsw_m": 48}, "guc": {}},
        "search_parameters": {"reloption": {"reset_me": None}, "guc": {"search_path": "foo,bar"}},
        "autotune_parameters": {
            "topk": [10, 100],
            "target_recall": [0.9, 0.95],
        },
    }
    config_path = tmp_path / "adbpg.yml"
    config_path.write_text(
        "adbpgnova:\n"
        "  db_label: cohere1m-novamr\n"
        "  user_name: tester\n"
        "  host: localhost\n"
        "  db_name: postgres\n"
        f"  build_parameters: {json.dumps(values['build_parameters'])}\n"
        f"  search_parameters: {json.dumps(values['search_parameters'])}\n"
        f"  autotune_parameters: {json.dumps(values['autotune_parameters'])}\n"
    )
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, ["--config-file", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output

    batch_args = build_sub_cmd_args(
        {
            "adbpgnova": [
                {
                    "db_label": "cohere1m-novamr",
                    "user_name": "tester",
                    "host": "localhost",
                    "db_name": "postgres",
                    "dry_run": True,
                    **values,
                }
            ]
        }
    )[0]
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, batch_args[1:])
    assert result.exit_code == 0, result.output
    assert captured[0][0].db_label == captured[1][0].db_label == "cohere1m-novamr"
    assert captured[0][1].model_dump(mode="json") == captured[1][1].model_dump(mode="json")


def test_cli_omits_autotune_to_disable_it(monkeypatch: pytest.MonkeyPatch):
    captured = []
    monkeypatch.setattr(adbpg_cli, "run", lambda **kwargs: captured.append(kwargs["db_case_config"]))

    result = CliRunner().invoke(
        adbpg_cli.AdbpgNova,
        ["--user-name", "tester", "--host", "localhost", "--db-name", "postgres"],
    )

    assert result.exit_code == 0, result.output
    assert captured[0].autotune_parameters is None

    batch_args = build_sub_cmd_args(
        {
            "adbpgnova": [
                {
                    "user_name": "tester",
                    "host": "localhost",
                    "db_name": "postgres",
                }
            ]
        }
    )[0]
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, batch_args[1:])
    assert result.exit_code == 0, result.output
    assert captured[1].autotune_parameters is None


def test_cli_accepts_formatted_multiline_json_parameters(monkeypatch: pytest.MonkeyPatch):
    captured = []
    monkeypatch.setattr(adbpg_cli, "run", lambda **kwargs: captured.append(kwargs["db_case_config"]))
    build_parameters = """{
  "reloption": {
    "algorithm": "novamr",
    "hnsw_m": 48,
    "hnsw_ef_construction": 600
  },
  "guc": {}
}"""
    search_parameters = """{
  "reloption": {},
  "guc": {
    "fastann.hnsw_ef_search": 130,
    "fastann.hnsw_max_scan_points": 5000,
    "fastann.quantize_rescore_amp": 2.0
  }
}"""

    result = CliRunner().invoke(
        adbpg_cli.AdbpgNova,
        [
            "--case-type",
            "Performance1024D1M",
            "--k",
            "10",
            "--host",
            "localhost",
            "--db-name",
            "postgres",
            "--user-name",
            "tester",
            "--password",
            "password",
            "--build-parameters",
            build_parameters,
            "--search-parameters",
            search_parameters,
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured[0]
    assert config.build_parameters.reloption["algorithm"] == "novamr"
    assert config.build_parameters.reloption["hnsw_m"] == 48
    assert config.build_parameters.reloption["hnsw_ef_construction"] == 600
    assert config.search_parameters.guc == {
        "fastann.hnsw_ef_search": 130,
        "fastann.hnsw_max_scan_points": 5000,
        "fastann.quantize_rescore_amp": 2.0,
    }


class TestResultRoundTrip:
    def test_read_file_rehydrates_structured_adbpg_config(self, tmp_path: Path):
        result_dir = tmp_path / "AnalyticDB for PostgreSQL"
        result_dir.mkdir()
        result_file = result_dir / "result_test_run.json"
        payload = {
            "run_id": "round-trip",
            "task_label": "round-trip",
            "results": [
                {
                    "metrics": {
                        "max_load_count": 0,
                        "insert_duration": 0.0,
                        "optimize_duration": 0.0,
                        "load_duration": 0.0,
                        "qps": 1.0,
                        "serial_latency_p99": 0.0,
                        "serial_latency_p95": 0.0,
                        "recall": 1.0,
                        "ndcg": 1.0,
                        "conc_num_list": [],
                        "conc_qps_list": [],
                        "conc_latency_p99_list": [],
                        "conc_latency_p95_list": [],
                        "conc_latency_avg_list": [],
                    },
                    "task_config": {
                        "db": DB.Adbpg.value,
                        "db_config": {"db_label": "", "version": "", "note": ""},
                        "db_case_config": make_index_config().model_dump(mode="json"),
                        "case_config": {"case_id": 5, "custom_case": {}, "k": 10},
                        "stages": ["search_serial"],
                        "load_concurrency": 0,
                    },
                    "label": ":)",
                }
            ],
            "timestamp": 0.0,
        }
        result_file.write_text(json.dumps(payload))

        result = TestResult.read_file(result_file, trans_unit=False)
        rehydrated = result.results[0].task_config.db_case_config
        assert isinstance(rehydrated, AdbpgIndexConfig)
        assert rehydrated.model_dump(mode="json") == make_index_config().model_dump(mode="json")
