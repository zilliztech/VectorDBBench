import json
import pickle
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from click.testing import CliRunner

pytest.importorskip("pymysql")

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.tidb import cli as tidb_cli
from vectordb_bench.backend.clients.tidb import spfresh as spfresh_module
from vectordb_bench.backend.clients.tidb.config import TiDBConfig, TiDBIndexConfig, TiDBSPFreshIndexConfig
from vectordb_bench.backend.clients.tidb.spfresh import TiDBSPFresh
from vectordb_bench.backend.clients.tidb.tidb import TiDB
from vectordb_bench.backend.filter import IntFilter, LabelFilter, non_filter
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend import task_runner as task_runner_module
from vectordb_bench.cli import cli as core_cli
from vectordb_bench.cli.batch_cli import build_sub_cmd_args
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, CaseType, TaskConfig, TestResult as BenchResult


def make_client(*, labels=False, mode="inline", metric=MetricType.COSINE, timeout=10, drop_old=False, **params):
    return TiDBSPFresh(
        dim=2,
        db_config={},
        db_case_config=TiDBSPFreshIndexConfig(
            metric_type=metric,
            spfresh_build_mode=mode,
            spfresh_wait_timeout=timeout,
            **params,
        ),
        with_scalar_labels=labels,
        drop_old=drop_old,
    )


def mock_connection(monkeypatch, client, rows=()):
    conn, cursor = Mock(), Mock()
    cursor.fetchone.side_effect = list(rows)

    @contextmanager
    def connect(*_args):
        yield conn, cursor

    monkeypatch.setattr(client, "_get_connection", connect)
    return conn, cursor


def ready(ts):
    return ts, "READY", 1, 0, None, "2026-11-17 00:00:00"


def spfresh_plan(index="idx_embedding_spfresh_cosine", operator="└─SPFreshVectorScan", table="vector_bench_test"):
    return [
        (operator, "3.00", "root", f"table:{table}, index:{index}(vec_cosine_distance(`embedding`))", "spfresh:COSINE")
    ]


def make_search_case(*, case_id=CaseType.Performance768D1M, custom=None, profile=PayloadProfile.IDS_ONLY):
    task = TaskConfig(
        db=DB.TiDBSPFresh,
        db_config=TiDBConfig(password=""),
        db_case_config=TiDBSPFreshIndexConfig(),
        case_config=CaseConfig(case_id=case_id, custom_case=custom or {}, payload_profile=profile, k=3),
        stages=["search_serial", "search_concurrent"],
    )
    runner = Assembler.assemble_all("test", "test", [task], DatasetSource.S3).case_runners[0]
    runner.ca.dataset.test_data = [[1.0, 0.0]]
    runner.ca.dataset.gt_data = [[7, 9, 11]]
    runner.init_db(drop_old=False)
    return runner


def fake_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(spfresh_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(spfresh_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def test_original_tidb_still_creates_tiflash_index_and_uses_tiflash_optimize(monkeypatch):
    client = TiDB(2, {}, TiDBIndexConfig(metric_type=MetricType.L2))
    _, cursor = mock_connection(monkeypatch, client)
    client._create_table()
    sql = cursor.execute.call_args.args[0]
    assert "VECTOR INDEX ((vec_l2_distance(embedding)))" in sql
    assert "SPFRESH" not in sql
    client.validate_search([1, 0], 100, non_filter, PayloadProfile.IDS_ONLY)
    cursor.execute.assert_called_once()
    hooks = {
        "_optimize_check_tiflash_replica_progress": 1,
        "_optimize_wait_tiflash_catch_up": 10,
        "_optimize_compact_tiflash": None,
        "_optimize_get_tiflash_index_pending_rows": 0,
    }
    for name, result in hooks.items():
        monkeypatch.setattr(client, name, Mock(return_value=result))
    client.optimize()
    for name in hooks:
        getattr(client, name).assert_called_once()
    assert DB.TiDB.init_cls is TiDB
    assert DB.TiDB.case_config_cls() is TiDBIndexConfig
    assert not TiDB.filter_supported(IntFilter(int_value=5))


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE])
@pytest.mark.parametrize("mode", ["inline", "non-inline"])
def test_spfresh_ddl_and_checkpoint_barrier(monkeypatch, metric, mode):
    param = (
        "min_partition_size=64,max_partition_size=512,write_beam_size=8,"
        "search_beam_size=174,oversample_factor=1.25,hamming_factor=96,rabitq4_factor=4"
    )
    conn, cursor = mock_connection(monkeypatch, TiDBSPFresh, [(100,), ready(99), ready(100)])
    clock = fake_clock(monkeypatch)
    client = make_client(
        labels=True,
        metric=metric,
        mode=mode,
        drop_old=True,
        min_partition_size=64,
        max_partition_size=512,
        write_beam_size=8,
        search_beam_size=174,
        oversample_factor=1.25,
        hamming_factor=96,
        rabitq4_factor=4,
    )
    create_sql, create_params = cursor.execute.call_args.args
    assert "labels VARCHAR(64) NOT NULL" in create_sql
    assert ("USING SPFRESH" in create_sql) == (mode == "inline")
    assert create_params == ((param,) if mode == "inline" else None)
    with client.init():
        assert client.optimize() is None
    calls = cursor.execute.call_args_list
    ddl = [call for call in calls if "USING SPFRESH" in call.args[0]]
    assert len(ddl) == 1
    assert "STORING (labels) VECTOR_INDEX_PARAM %s" in ddl[0].args[0]
    assert f"idx_embedding_spfresh_{metric.value.lower()}" in ddl[0].args[0]
    assert ddl[0].args[1] == (param,)
    assert not any("ALTER INDEX" in call.args[0] for call in calls)
    assert not any("DROP INDEX" in call.args[0] or "tiflash" in call.args[0].lower() for call in calls)
    assert clock[0] == 1.0
    assert conn.commit.call_count == (4 if mode == "non-inline" else 3)


def test_omitted_parameters_use_server_defaults_on_creation(monkeypatch):
    _, cursor = mock_connection(monkeypatch, TiDBSPFresh)
    make_client(drop_old=True)
    sql, args = cursor.execute.call_args.args
    assert "USING SPFRESH" in sql
    assert "VECTOR_INDEX_PARAM" not in sql
    assert args is None


def test_reuse_without_search_parameters_does_not_touch_database(monkeypatch):
    connect = Mock(side_effect=AssertionError("No SQL expected when keeping existing search settings"))
    monkeypatch.setattr(TiDBSPFresh, "_get_connection", connect)
    make_client(min_partition_size=64, max_partition_size=512, write_beam_size=8)
    connect.assert_not_called()


def test_search_parameter_update_failure_aborts_initialization(monkeypatch):
    conn, cursor = mock_connection(monkeypatch, TiDBSPFresh)
    cursor.execute.side_effect = RuntimeError("ALTER failed")
    with pytest.raises(RuntimeError, match="ALTER failed"):
        make_client(search_beam_size=174)
    cursor.execute.assert_called_once()
    conn.commit.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        None,
        (1,),
        (None, "READY", 1, None, None, "now"),
        (0, "READY", 1, 0, None, "now"),
        (100, "NEEDS_REBUILD", 0, None, None, "now"),
        (100, "UNKNOWN", 0, None, None, "now"),
        (100, "READY", 0, 0, None, "now"),
        (100, "READY", 1, 0, None, None),
    ],
)
def test_invalid_status_fails_before_search(monkeypatch, status):
    client = make_client()
    _, cursor = mock_connection(monkeypatch, client, [(100,), status])
    fake_clock(monkeypatch)
    with client.init(), pytest.raises(RuntimeError, match="SPFresh.*status|SPFresh index is not ready"):
        client.optimize()
    assert cursor.fetchone.call_count == 2


def test_checkpoint_timeout_includes_last_status_and_bounds_sleep(monkeypatch):
    client = make_client(timeout=0.25)
    mock_connection(monkeypatch, client, [(100,), ready(99)])
    clock = fake_clock(monkeypatch)
    with client.init(), pytest.raises(RuntimeError, match="Timed out.*barrier_ts=100.*last_status=.*99"):
        client.optimize()
    assert clock[0] == 0.25


def test_checkpoint_observed_after_deadline_is_rejected(monkeypatch):
    client = make_client(timeout=1)
    _, cursor = mock_connection(monkeypatch, client)
    clock = fake_clock(monkeypatch)

    def fetch():
        if cursor.fetchone.call_count == 1:
            return (100,)
        clock[0] = 1
        return ready(101)

    cursor.fetchone.side_effect = fetch
    with client.init(), pytest.raises(RuntimeError, match="Timed out"):
        client.optimize()


def test_status_query_error_preserves_cause(monkeypatch):
    client = make_client()
    error = RuntimeError("status table unavailable")
    mock_connection(monkeypatch, client, [(100,), error])
    with client.init(), pytest.raises(RuntimeError, match="SPFresh status query failed") as exc:
        client.optimize()
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("row", [None, (None,), (0,)])
def test_missing_tso_is_rejected(monkeypatch, row):
    client = make_client()
    mock_connection(monkeypatch, client, [row])
    with client.init(), pytest.raises(RuntimeError, match="positive TSO"):
        client.optimize()


@pytest.mark.parametrize("count", [0, 1, 9, 11, 31])
@pytest.mark.parametrize("labels", [False, True])
def test_insert_batches_cover_all_rows_and_bind_labels(monkeypatch, count, labels):
    client = make_client(labels=labels)
    conn, cursor = mock_connection(monkeypatch, client)
    values = ["label'\\value"] * count if labels else None
    assert client.insert_embeddings([[1.0, 0.0]] * count, list(range(count)), labels_data=values) == (count, None)
    calls = cursor.execute.call_args_list
    if labels:
        ids = sorted(row_id for call in calls for row_id in call.args[1][::3])
        assert ids == list(range(count))
        assert all(value == "label'\\value" for call in calls for value in call.args[1][2::3])
        assert all("label'\\value" not in call.args[0] for call in calls)
    else:
        assert sum(call.args[0].count("[1.0, 0.0]") for call in calls) == count
    assert conn.commit.call_count == len(calls)


def test_missing_labels_and_write_errors_propagate(monkeypatch):
    client = make_client(labels=True)
    with pytest.raises(ValueError, match="one label per embedding"):
        client.insert_embeddings([[1, 0]], [1])
    _, cursor = mock_connection(monkeypatch, client)
    cursor.execute.side_effect = RuntimeError("write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        client.insert_embeddings([[1, 0]], [1], labels_data=["label"])


def test_standard_concurrent_runner_loads_labels_without_exporting_state(monkeypatch):
    client = make_client(labels=True)
    _, cursor = mock_connection(monkeypatch, client)
    dataset = SimpleNamespace(
        data=SimpleNamespace(train_id_field="id", train_vector_field="emb", scalar_labels_file_separated=False),
        iter_batches=lambda size: iter(
            [pd.DataFrame({"id": [i], "emb": [[1.0, 0.0]], "labels": ["label_1p"]}) for i in range(4)]
        ),
    )
    runner = ConcurrentInsertRunner(
        client, dataset, normalize=False, max_workers=2, batch_size=1, with_scalar_labels=True
    )
    assert runner.max_workers == 2
    assert runner.task() == 4
    assert sorted(call.args[1][0] for call in cursor.execute.call_args_list) == [0, 1, 2, 3]
    # optimize runs in a separate process: it must not need the loader's client state.
    optimizer = make_client(labels=True)
    mock_connection(monkeypatch, optimizer, [(200,), ready(201)])
    with optimizer.init():
        optimizer.optimize()


@pytest.mark.parametrize(
    "filters,expected",
    [(non_filter, ()), (IntFilter(int_value=42), (42,)), (LabelFilter(label_percentage=0.01), ("label_1p",))],
)
def test_search_binds_filters_and_can_reset_them(filters, expected):
    client = make_client(labels=True)
    client.cursor = Mock()
    client.cursor.fetchall.return_value = [("7",), (9,)]
    client.prepare_filter(filters)
    assert client.search_embedding([1, 0], k=2) == [7, 9]
    sql, params = client.cursor.execute.call_args.args
    assert params == (*expected, "[1, 0]", 2)
    assert ("WHERE" in sql) == bool(expected)
    client.prepare_filter(non_filter)
    client.search_embedding([1, 0])
    assert "WHERE" not in client.cursor.execute.call_args.args[0]


@pytest.mark.parametrize("profile", [PayloadProfile.IDS_ONLY, PayloadProfile.SCALAR_LABEL])
@pytest.mark.parametrize(
    "case_id,custom",
    [
        (CaseType.Performance768D1M, {}),
        (
            CaseType.NewIntFilterPerformanceCase,
            {"dataset_with_size_type": "Medium Cohere (768dim, 1M)", "filter_rate": 0.99},
        ),
        (
            CaseType.LabelFilterPerformanceCase,
            {"dataset_with_size_type": "Medium Cohere (768dim, 1M)", "label_percentage": 0.01},
        ),
    ],
)
def test_search_preflight_and_payload_use_actual_case_query(monkeypatch, profile, case_id, custom):
    runner = make_search_case(case_id=case_id, custom=custom, profile=profile)
    _, cursor = mock_connection(monkeypatch, runner.db, [ready(100)])
    cursor.fetchall.return_value = spfresh_plan()
    runner._init_search_runners()
    status_call, explain_call = cursor.execute.call_args_list
    assert status_call.args[1] == ("vector_bench_test", "idx_embedding_spfresh_cosine")
    sql, params = explain_call.args
    columns = "id, labels" if profile == PayloadProfile.SCALAR_LABEL else "id"
    assert sql.startswith(f"EXPLAIN FORMAT = 'brief' SELECT {columns} FROM ")
    assert params[-2:] == ("[1.0, 0.0]", 3)
    if case_id == CaseType.LabelFilterPerformanceCase:
        assert "WHERE labels = %s" in sql
        assert params[0] == "label_1p"
    elif case_id == CaseType.NewIntFilterPerformanceCase:
        assert "WHERE id >= %s" in sql
        assert params[0] == runner.ca.filters.int_value
    else:
        assert "WHERE" not in sql
    cursor.fetchall.return_value = (
        [(7, "label_1p"), (9, "label_10p")] if profile == PayloadProfile.SCALAR_LABEL else [(7,), (9,)]
    )
    # Both standard runners request the selected payload; only IDs enter recall scoring.
    for search_runner in (runner.serial_search_runner, runner.search_runner):
        with runner.db.init():
            assert search_runner._search_embedding([1.0, 0.0]) == [7, 9]
        assert cursor.execute.call_args.args == (sql.removeprefix("EXPLAIN FORMAT = 'brief' "), params)
    assert cursor.execute.call_count == 4  # status + EXPLAIN + two measured queries


@pytest.mark.parametrize(
    "status,plan,error",
    [
        (None, [], "Missing or invalid SPFresh status"),
        ((100, "NEEDS_REBUILD", 0, 0, None, "now"), [], "not ready"),
        (
            ready(100),
            [("TableFullScan", "3", "cop[tikv]", "table:vector_bench_test", "keep order:false")],
            "does not use",
        ),
        (ready(100), spfresh_plan(index="idx_embedding_spfresh_cosine_other"), "does not use"),
        (ready(100), spfresh_plan(table="other_table"), "does not use"),
        (ready(100), spfresh_plan(operator="TableFullScan"), "does not use"),
        (ready(100), [], "does not use"),
        (ready(100), RuntimeError("EXPLAIN failed"), "EXPLAIN failed"),
    ],
)
def test_search_only_fails_before_search_runners_start(monkeypatch, status, plan, error):
    runner = make_search_case()
    _, cursor = mock_connection(monkeypatch, runner.db, [status])
    if isinstance(plan, Exception):
        cursor.fetchall.side_effect = plan
    else:
        cursor.fetchall.return_value = plan
    serial, concurrent = Mock(), Mock()
    monkeypatch.setattr(task_runner_module, "SerialSearchRunner", serial)
    monkeypatch.setattr(task_runner_module, "MultiProcessingSearchRunner", concurrent)
    with pytest.raises(RuntimeError, match=error):
        runner._init_search_runners()
    serial.assert_not_called()
    concurrent.assert_not_called()
    assert not any(call.args[0].startswith("SELECT id") for call in cursor.execute.call_args_list)


def test_scalar_label_payload_requires_labels_and_rejects_other_profiles():
    client = make_client()
    client.cursor = Mock()
    with pytest.raises(ValueError, match="requires a table with scalar labels"):
        client.search_embedding([1, 0], payload_profile=PayloadProfile.SCALAR_LABEL)
    with pytest.raises(ValueError, match="Unsupported.*payload"):
        client.search_embedding([1, 0], payload_profile=PayloadProfile.VECTOR)
    client.cursor.execute.assert_not_called()


def test_cold_warm_search_also_validates_before_runner_creation(monkeypatch):
    runner = make_search_case(
        case_id=CaseType.CloudColdLatencyCase,
        custom={"dataset_with_size_type": "Medium Cohere (768dim, 1M)", "query_count": 1},
        profile=None,
    )
    mock_connection(monkeypatch, runner.db, [None])
    cold_warm = Mock()
    monkeypatch.setattr(task_runner_module, "ColdWarmSearchRunner", cold_warm)
    with pytest.raises(RuntimeError, match="Missing or invalid SPFresh status"):
        runner._init_cold_warm_search_runner()
    cold_warm.assert_not_called()


@pytest.mark.parametrize("filters", [IntFilter(int_field="other", int_value=1), LabelFilter(label_percentage=0.01)])
def test_unsupported_filter_schema_is_rejected(filters):
    with pytest.raises(ValueError, match="Unsupported"):
        make_client().prepare_filter(filters)


@pytest.mark.parametrize("command,db", [(tidb_cli.TiDB, DB.TiDB), (tidb_cli.TiDBSPFresh, DB.TiDBSPFresh)])
def test_cli_preserves_tidb_defaults_and_requires_explicit_spfresh(monkeypatch, command, db):
    run = Mock()
    monkeypatch.setattr(tidb_cli, "run", run)
    result = CliRunner().invoke(command, ["--dry-run"])
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["db"] == db
    assert type(run.call_args.kwargs["db_case_config"]) is db.case_config_cls()
    assert run.call_args.kwargs["db_config"].db_name == "test"


def test_batch_cli_passes_spfresh_parameters(monkeypatch):
    run = Mock()
    monkeypatch.setattr(tidb_cli, "run", run)
    args = build_sub_cmd_args(
        {
            "tidbspfresh": [
                {
                    "db_name": "cohere_1m",
                    "spfresh_build_mode": "non-inline",
                    "min_partition_size": 64,
                    "max_partition_size": 512,
                    "write_beam_size": 8,
                    "search_beam_size": 174,
                    "oversample_factor": 1.25,
                    "hamming_factor": 96,
                    "rabitq4_factor": 4,
                    "spfresh_wait_timeout": 30,
                    "dry_run": True,
                }
            ]
        }
    )[0]
    result = CliRunner().invoke(tidb_cli.TiDBSPFresh, args[1:])
    assert result.exit_code == 0, result.output
    config = run.call_args.kwargs["db_case_config"]
    assert config.spfresh_build_mode == "non-inline"
    assert (config.min_partition_size, config.max_partition_size, config.write_beam_size) == (64, 512, 8)
    assert (config.search_beam_size, config.oversample_factor, config.hamming_factor, config.rabitq4_factor) == (
        174,
        1.25,
        96,
        4,
    )
    assert config.spfresh_wait_timeout == 30


@pytest.mark.parametrize(
    "search_options,expected_params",
    [
        (["--search-beam-size", "174"], "search_beam_size=174"),
        (
            [
                "--search-beam-size",
                "256",
                "--oversample-factor",
                "2",
                "--hamming-factor",
                "64",
                "--rabitq4-factor",
                "3",
            ],
            "search_beam_size=256,oversample_factor=2.0,hamming_factor=64,rabitq4_factor=3",
        ),
    ],
)
def test_search_only_updates_parameters_once_before_workers(monkeypatch, search_options, expected_params):
    tasks = []

    def capture_task(**kwargs):
        task = TaskConfig(**kwargs)
        tasks.append(task)
        return task

    monkeypatch.setattr(core_cli, "TaskConfig", capture_task)
    result = CliRunner().invoke(
        tidb_cli.TiDBSPFresh,
        [
            "--skip-load",
            "--dry-run",
            "--case-type",
            "Performance768D1M",
            "--k",
            "100",
            "--min-partition-size",
            "64",
            "--max-partition-size",
            "512",
            "--write-beam-size",
            "8",
            *search_options,
        ],
    )
    assert result.exit_code == 0, result.output
    assert tasks[0].stages == ["search_serial", "search_concurrent"]
    assert tasks[0].case_config.k == 100
    conn, cursor = mock_connection(monkeypatch, TiDBSPFresh)
    cursor.fetchall.return_value = [(7,)]
    runner = Assembler.assemble_all("test", "test", tasks, DatasetSource.S3).case_runners[0]
    runner.init_db(drop_old=False)
    cursor.execute.assert_called_once_with(
        "ALTER TABLE vector_bench_test ALTER INDEX idx_embedding_spfresh_cosine SET VECTOR_INDEX_PARAM %s",
        (expected_params,),
    )
    conn.commit.assert_called_once()
    # Workers receive a copy of the initialized client, then open their own sessions.
    for _ in range(2):
        worker_client = pickle.loads(pickle.dumps(runner.db))
        with worker_client.init():
            worker_client.prepare_filter(non_filter)
            assert worker_client.search_embedding([1, 0]) == [7]
    calls = [call.args[0] for call in cursor.execute.call_args_list]
    assert len(calls) == 3
    assert all(sql.startswith("SELECT id") for sql in calls[1:])


def test_original_command_does_not_accept_spfresh_options():
    result = CliRunner().invoke(tidb_cli.TiDB, ["--spfresh-build-mode", "inline", "--dry-run"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "case_id,custom",
    [
        (CaseType.Performance768D1M, {}),
        (
            CaseType.NewIntFilterPerformanceCase,
            {"dataset_with_size_type": "Medium Cohere (768dim, 1M)", "filter_rate": 0.99},
        ),
        (
            CaseType.LabelFilterPerformanceCase,
            {"dataset_with_size_type": "Medium Cohere (768dim, 1M)", "label_percentage": 0.01},
        ),
    ],
)
def test_existing_cases_assemble_and_initialize_spfresh_with_standard_runner(case_id, custom):
    task = TaskConfig(
        db=DB.TiDBSPFresh,
        db_config=TiDBConfig(password=""),
        db_case_config=TiDBSPFreshIndexConfig(),
        case_config=CaseConfig(case_id=case_id, custom_case=custom, k=100),
    )
    runner = Assembler.assemble_all("test", "test", [task], DatasetSource.S3).case_runners[0]
    runner.init_db(drop_old=False)
    assert isinstance(runner.db, TiDBSPFresh)
    assert runner.db.case_config.metric_type == MetricType.COSINE
    assert runner.db.with_scalar_labels == (case_id == CaseType.LabelFilterPerformanceCase)
    runner.db.prepare_filter(runner.ca.filters)


@pytest.mark.parametrize(
    "args",
    [
        ["--spfresh-build-mode", "split"],
        ["--spfresh-wait-timeout", "0"],
        ["--min-partition-size", "0"],
        ["--max-partition-size", "1"],
        ["--write-beam-size", "4097"],
        ["--search-beam-size", "0"],
        ["--oversample-factor", "0.5"],
        ["--hamming-factor", "257"],
        ["--rabitq4-factor", "0"],
    ],
)
def test_cli_rejects_unsupported_mode_and_invalid_parameters(args):
    result = CliRunner().invoke(tidb_cli.TiDBSPFresh, ["--dry-run", *args])
    assert result.exit_code == 2


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_config_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        TiDBSPFreshIndexConfig(spfresh_wait_timeout=timeout)


@pytest.mark.parametrize(
    "params",
    [
        {"min_partition_size": 0},
        {"max_partition_size": 1},
        {"write_beam_size": 0},
        {"search_beam_size": 4097},
        {"oversample_factor": float("nan")},
        {"oversample_factor": float("inf")},
        {"oversample_factor": 17},
        {"hamming_factor": 0},
        {"rabitq4_factor": 257},
        {"min_partition_size": 256, "max_partition_size": 64},
        {"hamming_factor": 3, "rabitq4_factor": 4},
    ],
)
def test_config_rejects_invalid_tuning_parameters(params):
    with pytest.raises(ValueError):
        TiDBSPFreshIndexConfig(**params)


def test_spfresh_result_round_trip_preserves_configuration(tmp_path):
    case_config = TiDBSPFreshIndexConfig(
        metric_type=MetricType.COSINE,
        spfresh_build_mode="non-inline",
        min_partition_size=64,
        search_beam_size=174,
        oversample_factor=1.25,
    )
    task = TaskConfig(
        db=DB.TiDBSPFresh,
        db_config=TiDBConfig(password=""),
        db_case_config=case_config,
        case_config=CaseConfig(case_id=CaseType.Performance768D1M),
    )
    result = BenchResult(
        run_id="spfresh-test", task_label="spfresh", results=[CaseResult(task_config=task, metrics=Metric())]
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result.model_dump_for_output()))
    restored = BenchResult.read_file(path).results[0].task_config
    assert restored.db == DB.TiDBSPFresh
    assert restored.db_case_config == case_config
    assert DB.TiDBSPFresh.init_cls is TiDBSPFresh
