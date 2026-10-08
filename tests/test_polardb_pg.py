from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.pgvector.pgvector import PgVector
from vectordb_bench.backend.clients.polardb_pg.cli import PolarDBPgHNSW as PolarDBPgHNSWCommand
from vectordb_bench.backend.clients.polardb_pg.config import PolarDBPgHNSWConfig
from vectordb_bench.backend.clients.polardb_pg.polardb_pg import (
    HNSWGraphCacheDetail,
    PolarDBPgHNSW,
)
from vectordb_bench.frontend.config.dbCaseConfigs import get_case_config_inputs
from vectordb_bench.models import CaseConfigParamType

if TYPE_CHECKING:
    from collections.abc import Iterator


def make_config(**kwargs) -> PolarDBPgHNSWConfig:
    return PolarDBPgHNSWConfig(
        metric_type="COSINE",
        m=16,
        ef_construction=256,
        ef_search=200,
        **kwargs,
    )


def cache_detail(
    status: str,
    *,
    requested: str = "on",
    usable: bool = False,
    last_error: str | None = None,
) -> HNSWGraphCacheDetail:
    detail = f"requested={requested} status={status} usable={'yes' if usable else 'no'}"
    if last_error is not None:
        detail += f" last_error={last_error}"
    return HNSWGraphCacheDetail.parse(detail)


def make_client(details: list[HNSWGraphCacheDetail], **config_kwargs) -> PolarDBPgHNSW:
    client = object.__new__(PolarDBPgHNSW)
    client.case_config = make_config(**config_kwargs)
    client._index_name = "pgvector_index"
    client._check_graph_cache_role = MagicMock()
    client._graph_cache_detail = MagicMock(side_effect=details)
    client._call_graph_cache_function = MagicMock(return_value=True)
    return client


class TestPolarDBPgConfig:
    def test_rabitq_graph_cache_index_options(self):
        config = make_config(
            quantization="rabitq",
            quantization_nbits=8,
            train_samples=100000,
        )

        options = {item["option_name"]: item["val"] for item in config.index_param()["index_creation_with_options"]}

        assert options == {
            "m": "16",
            "ef_construction": "256",
            "quantization": "rabitq",
            "train_samples": "100000",
            "quantization_nbits": "8",
            "cache": "on",
        }

    def test_frontend_quantization_key_maps_to_index_option(self):
        config = make_config(
            hnsw_quantization="rabitq",
            quantization_nbits=8,
        )

        options = {item["option_name"]: item["val"] for item in config.index_param()["index_creation_with_options"]}

        assert config.quantization == "rabitq"
        assert options["quantization"] == "rabitq"
        assert options["quantization_nbits"] == "8"

    def test_pq_index_options(self):
        config = make_config(
            graph_cache=False,
            quantization="pq",
            pq_m=32,
            train_samples=5000,
        )

        options = {item["option_name"]: item["val"] for item in config.index_param()["index_creation_with_options"]}

        assert options == {
            "m": "16",
            "ef_construction": "256",
            "quantization": "pq",
            "pq_m": "32",
            "train_samples": "5000",
        }

    @pytest.mark.parametrize("quantization", ["sq4", "sq8"])
    def test_sq_index_options(self, quantization: str):
        config = make_config(
            graph_cache=False,
            quantization=quantization,
            train_samples=5000,
        )

        options = {item["option_name"]: item["val"] for item in config.index_param()["index_creation_with_options"]}

        assert options == {
            "m": "16",
            "ef_construction": "256",
            "quantization": quantization,
            "train_samples": "5000",
        }

    def test_graph_cache_can_be_disabled(self):
        config = make_config(graph_cache=False, iterative_scan="relaxed_order")
        options = {item["option_name"]: item["val"] for item in config.index_param()["index_creation_with_options"]}
        assert "cache" not in options

    def test_graph_cache_timeout_must_be_positive(self):
        assert make_config().graph_cache_timeout == 3600
        with pytest.raises(ValidationError):
            make_config(graph_cache_timeout=0)

    @pytest.mark.parametrize("nbits", [1, 4, 8])
    def test_supported_rabitq_nbits(self, nbits: int):
        assert make_config(quantization="rabitq", quantization_nbits=nbits).quantization_nbits == nbits

    @pytest.mark.parametrize("nbits", [0, 2, 9])
    def test_rejects_unsupported_rabitq_nbits(self, nbits: int):
        with pytest.raises(ValidationError):
            make_config(quantization="rabitq", quantization_nbits=nbits)

    def test_rejects_iterative_scan_with_graph_cache(self):
        with pytest.raises(ValidationError, match="Graph Cache requires"):
            make_config(iterative_scan="strict_order")

    def test_frontend_post_load_switch_controls_index_timing(self):
        before_load = make_config(post_load_index=False)
        after_load = make_config(post_load_index=True)

        assert before_load.create_index_before_load is True
        assert before_load.create_index_after_load is False
        assert after_load.create_index_before_load is False
        assert after_load.create_index_after_load is True

    def test_rejects_quantizer_specific_options(self):
        with pytest.raises(ValidationError, match="pq_m"):
            make_config(quantization="sq8", pq_m=16)
        with pytest.raises(ValidationError, match="quantization_nbits"):
            make_config(quantization="pq", quantization_nbits=8)

    def test_rejects_opq(self):
        with pytest.raises(ValidationError):
            make_config(quantization="opq")

    def test_connection_password_is_redacted(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        connection = MagicMock()
        cursor = MagicMock()
        monkeypatch.setattr(PgVector, "_create_connection", MagicMock(return_value=(connection, cursor)))
        caplog.set_level("INFO", logger="vectordb_bench.backend.clients.pgvector.pgvector")

        PgVector(
            dim=8,
            db_config={
                "connect_config": {
                    "host": "localhost",
                    "port": 5432,
                    "dbname": "vectordb",
                    "user": "postgres",
                    "password": "do-not-log-this-password",
                },
                "table_name": "vdbbench_table_test",
            },
            db_case_config=make_config(graph_cache=False),
        )

        assert "do-not-log-this-password" not in caplog.text
        assert "**********" in caplog.text

    def test_polardb_name_is_used_during_base_initialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        connection = MagicMock()
        cursor = MagicMock()
        monkeypatch.setattr(PgVector, "_create_connection", MagicMock(return_value=(connection, cursor)))
        caplog.set_level("INFO", logger="vectordb_bench.backend.clients.pgvector.pgvector")

        client = PolarDBPgHNSW(
            dim=8,
            db_config={
                "connect_config": {
                    "host": "localhost",
                    "port": 5432,
                    "dbname": "vectordb",
                    "user": "postgres",
                    "password": "secret",
                },
                "table_name": "vdbbench_table_test",
            },
            db_case_config=make_config(graph_cache=False),
        )

        assert client.name == "PolarDBPG"
        assert "PolarDBPG config values" in caplog.text
        assert "PgVector config values" not in caplog.text


class TestGraphCacheDetail:
    def test_parse_ready_detail(self):
        detail = HNSWGraphCacheDetail.parse(
            "requested=on dbid=1 index_oid=2 status=ready usable=yes graph_bytes=100 last_error=none",
        )
        assert detail.requested == "on"
        assert detail.status == "ready"
        assert detail.usable is True
        assert detail.last_error is None

    def test_parse_error_with_spaces(self):
        detail = HNSWGraphCacheDetail.parse(
            "requested=on status=empty usable=no last_error=could not read graph page",
        )
        assert detail.last_error == "could not read graph page"

    def test_rejects_incomplete_detail(self):
        with pytest.raises(RuntimeError, match="missing usable"):
            HNSWGraphCacheDetail.parse("requested=on status=empty")


class TestGraphCacheLifecycle:
    def test_search_only_initialization_waits_for_cache(self, monkeypatch: pytest.MonkeyPatch):
        def base_init(
            client: PolarDBPgHNSW,
            *_args: object,
            db_case_config: PolarDBPgHNSWConfig,
            **_kwargs: object,
        ) -> None:
            client.case_config = db_case_config
            client._index_name = "pgvector_index"

        @contextmanager
        def fake_init(_client: PolarDBPgHNSW) -> Iterator[None]:
            yield

        wait_for_cache = MagicMock()
        monkeypatch.setattr(PgVector, "__init__", base_init)
        monkeypatch.setattr(PolarDBPgHNSW, "init", fake_init)
        monkeypatch.setattr(PolarDBPgHNSW, "_ensure_graph_cache_ready", wait_for_cache)

        PolarDBPgHNSW(db_case_config=make_config(), drop_old=False)

        wait_for_cache.assert_called_once_with()

    def test_post_insert_waits_after_index_creation(self, monkeypatch: pytest.MonkeyPatch):
        create_index = MagicMock()
        wait_for_cache = MagicMock()
        client = object.__new__(PolarDBPgHNSW)
        client.case_config = make_config()
        monkeypatch.setattr(PgVector, "_post_insert", create_index)
        monkeypatch.setattr(PolarDBPgHNSW, "_ensure_graph_cache_ready", wait_for_cache)

        client._post_insert()

        create_index.assert_called_once_with()
        wait_for_cache.assert_called_once_with()

    def test_ready_cache_needs_no_lifecycle_call(self):
        client = make_client([cache_detail("ready", usable=True)])

        client._ensure_graph_cache_ready()

        client._check_graph_cache_role.assert_called_once_with()
        client._call_graph_cache_function.assert_not_called()

    def test_empty_cache_is_scheduled_before_ready(self, monkeypatch: pytest.MonkeyPatch):
        client = make_client(
            [
                cache_detail("empty"),
                cache_detail("building"),
                cache_detail("ready", usable=True),
            ],
        )
        monkeypatch.setattr("vectordb_bench.backend.clients.polardb_pg.polardb_pg.time.sleep", lambda _: None)

        client._ensure_graph_cache_ready()

        client._call_graph_cache_function.assert_called_once_with("hnsw_schedule_cache_rebuild")

    def test_stale_cache_is_released_and_rebuilt(self, monkeypatch: pytest.MonkeyPatch):
        client = make_client(
            [
                cache_detail("ready", usable=False),
                cache_detail("draining"),
                cache_detail("empty"),
                cache_detail("building"),
                cache_detail("ready", usable=True),
            ],
        )
        monkeypatch.setattr("vectordb_bench.backend.clients.polardb_pg.polardb_pg.time.sleep", lambda _: None)

        client._ensure_graph_cache_ready()

        assert client._call_graph_cache_function.call_args_list == [
            call("hnsw_release_cache"),
            call("hnsw_schedule_cache_rebuild"),
        ]

    def test_not_preloaded_fails_immediately(self):
        client = make_client([cache_detail("not_preloaded")])
        with pytest.raises(RuntimeError, match="shared_preload_libraries"):
            client._ensure_graph_cache_ready()

    def test_index_without_cache_reloption_fails(self):
        client = make_client([cache_detail("empty", requested="off")])
        with pytest.raises(RuntimeError, match="cache=on"):
            client._ensure_graph_cache_ready()

    def test_build_error_fails_without_retry(self):
        client = make_client([cache_detail("empty", last_error="out of shared memory")])
        with pytest.raises(RuntimeError, match="out of shared memory"):
            client._ensure_graph_cache_ready()
        client._call_graph_cache_function.assert_not_called()

    def test_timeout_reports_last_state(self, monkeypatch: pytest.MonkeyPatch):
        client = make_client([cache_detail("building")], graph_cache_timeout=1)
        monotonic = MagicMock(side_effect=[0.0, 2.0])
        monkeypatch.setattr("vectordb_bench.backend.clients.polardb_pg.polardb_pg.time.monotonic", monotonic)

        with pytest.raises(TimeoutError, match="after 1s.*status=building"):
            client._ensure_graph_cache_ready()


def test_polardb_pg_cli_help():
    result = CliRunner().invoke(PolarDBPgHNSWCommand, ["--help"])

    assert result.exit_code == 0, result.output
    assert PolarDBPgHNSWCommand.name == "polardbpghnsw"
    assert "--graph-cache / --skip-graph-cache" in result.output
    assert "--graph-cache-poll-interval" not in result.output
    assert "--graph-cache-timeout INTEGER" in result.output
    assert "--quantization [pq|sq4|sq8|rabitq]" in result.output
    assert "--quantization-nbits [1|4|8]" in result.output


def test_polardb_pg_cli_dry_run_builds_task(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    benchmark_run = MagicMock()
    monkeypatch.setattr("vectordb_bench.cli.cli.benchmark_runner.run", benchmark_run)
    caplog.set_level("INFO", logger="vectordb_bench.cli.cli")

    result = CliRunner().invoke(
        PolarDBPgHNSWCommand,
        [
            "--case-type",
            "Performance1024D1M",
            "--db-label",
            "polardb-pg-dry-run",
            "--user-name",
            "postgres",
            "--password",
            "secret",
            "--host",
            "localhost",
            "--port",
            "5432",
            "--db-name",
            "vec",
            "--m",
            "16",
            "--ef-construction",
            "256",
            "--ef-search",
            "200",
            "--maintenance-work-mem",
            "128GB",
            "--max-parallel-workers",
            "64",
            "--quantization-type",
            "none",
            "--table-quantization-type",
            "none",
            "--skip-reranking",
            "--quantized-fetch-limit",
            "400",
            "--iterative-scan",
            "off",
            "--skip-create-index-before-load",
            "--create-index-after-load",
            "--graph-cache",
            "--graph-cache-timeout",
            "120",
            "--quantization",
            "rabitq",
            "--train-samples",
            "5000",
            "--quantization-nbits",
            "8",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    benchmark_run.assert_not_called()
    assert "graph_cache_timeout=120" in caplog.text
    assert "quantization='rabitq'" in caplog.text
    assert "secret" not in caplog.text


class TestPolarDBPgFrontend:
    @staticmethod
    def input_map(case_label: CaseLabel):
        return {config.label: config for config in get_case_config_inputs(DB.PolarDBPG, case_label)}

    def test_uses_hnsw_only_and_exposes_polardb_options(self):
        load_inputs = self.input_map(CaseLabel.Load)
        performance_inputs = self.input_map(CaseLabel.Performance)

        assert load_inputs[CaseConfigParamType.IndexType].inputConfig["options"] == [IndexType.HNSW.value]
        assert CaseConfigParamType.hnsw_quantization in load_inputs
        assert CaseConfigParamType.quantization_nbits in load_inputs
        assert CaseConfigParamType.graph_cache in load_inputs
        assert CaseConfigParamType.graph_cache_timeout in load_inputs
        assert CaseConfigParamType.post_load_index in load_inputs
        assert CaseConfigParamType.ef_search in performance_inputs
        assert CaseConfigParamType.graph_cache_timeout in performance_inputs
        assert CaseConfigParamType.iterative_scan in performance_inputs

    def test_quantization_parameter_has_a_distinct_enum_value(self):
        assert CaseConfigParamType.hnsw_quantization.value == "hnsw_quantization"
        assert CaseConfigParamType.hnsw_quantization is not CaseConfigParamType.mongodb_quantization_type

    def test_quantizer_fields_are_conditionally_displayed(self):
        inputs = self.input_map(CaseLabel.Performance)
        config = {
            CaseConfigParamType.IndexType: IndexType.HNSW.value,
            CaseConfigParamType.hnsw_quantization: "rabitq",
            CaseConfigParamType.graph_cache: True,
        }

        assert inputs[CaseConfigParamType.train_samples].isDisplayed(config)
        assert inputs[CaseConfigParamType.quantization_nbits].isDisplayed(config)
        assert not inputs[CaseConfigParamType.pq_m].isDisplayed(config)

        config[CaseConfigParamType.hnsw_quantization] = "pq"
        assert inputs[CaseConfigParamType.pq_m].isDisplayed(config)
        assert not inputs[CaseConfigParamType.quantization_nbits].isDisplayed(config)

    def test_iterative_scan_is_hidden_with_graph_cache(self):
        inputs = self.input_map(CaseLabel.Performance)
        config = {CaseConfigParamType.graph_cache: True}

        assert inputs[CaseConfigParamType.graph_cache_timeout].isDisplayed(config)
        assert not inputs[CaseConfigParamType.iterative_scan].isDisplayed(config)

        config[CaseConfigParamType.graph_cache] = False
        assert not inputs[CaseConfigParamType.graph_cache_timeout].isDisplayed(config)
        assert inputs[CaseConfigParamType.iterative_scan].isDisplayed(config)
