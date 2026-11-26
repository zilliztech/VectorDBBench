"""Offline tests for the PgSVS config and client layer.

These tests do not require a live database -- the connection is mocked out
entirely. For an end-to-end test against a real SVS-enabled Postgres, build
the extension per docs/build_guide/ in the SVS-Extension-for-PostgreSQL repo
(unlike plain pgvector, there is no prebuilt "svs" Docker image to run against).

Usage:
  pytest tests/test_pgsvs.py -v
"""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, call

import pytest
from pydantic import ValidationError

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.pgsvs.config import PgSVSVamanaConfig
from vectordb_bench.backend.filter import IntFilter, non_filter

if TYPE_CHECKING:
    from vectordb_bench.backend.clients.pgsvs.pgsvs import PgSVS

DB_CONFIG = {
    "connect_config": {
        "host": "localhost",
        "port": 5432,
        "dbname": "vectordb",
        "user": "vectordb",
        "password": "super-secret-pw",
    },
    "table_name": "test_pgsvs",
}

DIM = 8


# -- Helpers ------------------------------------------------------------------


def make_config(**overrides) -> PgSVSVamanaConfig:
    values = {"metric_type": MetricType.COSINE}
    values.update(overrides)
    return PgSVSVamanaConfig(**values)


def pgsvs_client_cls():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from vectordb_bench.backend.clients.pgsvs.pgsvs import PgSVS  # noqa: PLC0415

    return PgSVS


def patch_sql_rendering(monkeypatch: pytest.MonkeyPatch):
    """Let sql.Composed.as_string() work without a real psycopg connection."""
    client_cls = pgsvs_client_cls()
    client_module = importlib.import_module(client_cls.__module__)
    original_as_string = client_module.sql.Composed.as_string
    monkeypatch.setattr(
        client_module.sql.Composed,
        "as_string",
        lambda composed, _context=None: original_as_string(composed),
    )
    return original_as_string


def make_db(
    table_name: str = "test_pgsvs",
    drop_old: bool = True,
    *,
    case_config: PgSVSVamanaConfig | None = None,
    with_scalar_labels: bool = False,
) -> PgSVS:
    config = dict(DB_CONFIG)
    config["connect_config"] = dict(DB_CONFIG["connect_config"])
    config["table_name"] = table_name
    return DB.PgSVS.init_cls(
        dim=DIM,
        db_config=config,
        db_case_config=case_config or make_config(),
        drop_old=drop_old,
        with_scalar_labels=with_scalar_labels,
    )


def _default_execute_side_effect(query: Any, *_args: Any, **_kwargs: Any) -> MagicMock:
    """pg_extension probe succeeds; anything else (e.g. the reuse-check query)
    reports "nothing found" by default, so drop_old=False skips cleanly unless
    a test overrides this to simulate a specific existing index."""
    result = MagicMock()
    if isinstance(query, str) and "pg_extension" in query:
        result.fetchone.return_value = (1,)
    else:
        result.fetchone.return_value = None
        result.fetchall.return_value = []
    return result


def existing_index_side_effect(row: tuple[str, str, str, list[str]]):
    """cursor.execute side_effect simulating a specific pre-existing index row
    for _validate_existing_index's catalog query, while still passing the
    pg_extension probe."""

    def side_effect(query: Any, *_args: Any, **_kwargs: Any) -> MagicMock:
        result = MagicMock()
        if isinstance(query, str) and "pg_extension" in query:
            result.fetchone.return_value = (1,)
        else:
            result.fetchone.return_value = row
        return result

    return side_effect


@pytest.fixture
def mocked_db_connection(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    client_cls = pgsvs_client_cls()
    conn = MagicMock(name="connection")
    cursor = MagicMock(name="cursor")
    cursor.execute.side_effect = _default_execute_side_effect
    monkeypatch.setattr(client_cls, "_create_connection", staticmethod(lambda **_kwargs: (conn, cursor)))
    return conn, cursor


@contextmanager
def capture_logger(name: str, caplog: pytest.LogCaptureFixture):
    """caplog relies on propagation to the root logger, but importing
    vectordb_bench runs log_util.init(), which sets propagate=False on the
    "vectordb_bench" logger tree -- so attach caplog's handler directly."""
    logger = logging.getLogger(name)
    logger.addHandler(caplog.handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)


def rendered_executed_sql(cursor: MagicMock) -> list[str]:
    """Render every statement passed to cursor.execute() as text, skipping
    bare strings (e.g. the pg_extension probe) which render as-is."""
    rendered = []
    for execute_call in cursor.execute.call_args_list:
        stmt = execute_call.args[0]
        rendered.append(stmt if isinstance(stmt, str) else stmt.as_string())
    return rendered


# -- Config validation: bounds and discrete values ---------------------------


class TestPgSVSConfigBounds:
    @pytest.mark.parametrize("value", ["bit"])
    @pytest.mark.parametrize("field", ["quantization_type", "table_quantization_type"])
    def test_rejects_bit_quantization(self, field: str, value: str) -> None:
        with pytest.raises(ValidationError, match="is not supported for svs"):
            make_config(**{field: value})

    @pytest.mark.parametrize("value", ["none", "halfvec", None])
    @pytest.mark.parametrize("field", ["quantization_type", "table_quantization_type"])
    def test_accepts_non_bit_quantization(self, field: str, value: str | None) -> None:
        make_config(**{field: value})  # must not raise

    @pytest.mark.parametrize("value", [16, 256, 100])
    def test_graph_degree_in_range(self, value: int) -> None:
        make_config(graph_degree=value)

    @pytest.mark.parametrize("value", [15, 257, 0, -1])
    def test_graph_degree_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError, match="graph_degree"):
            make_config(graph_degree=value)

    @pytest.mark.parametrize("value", [-1, 0, 200])
    def test_alpha_in_range(self, value: int) -> None:
        make_config(alpha=value)

    @pytest.mark.parametrize("value", [-2, 201])
    def test_alpha_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError, match="alpha"):
            make_config(alpha=value)

    @pytest.mark.parametrize("value", [-1, 0, 1000])
    def test_build_window_size_in_range(self, value: int) -> None:
        make_config(build_window_size=value)

    @pytest.mark.parametrize("value", [-2, 1001])
    def test_build_window_size_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError, match="build_window_size"):
            make_config(build_window_size=value)

    @pytest.mark.parametrize("value", [10, 10000])
    def test_search_window_size_in_range(self, value: int) -> None:
        make_config(search_window_size=value)

    @pytest.mark.parametrize("value", [9, 10001, -1])
    def test_search_window_size_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError, match="search_window_size"):
            make_config(search_window_size=value)

    @pytest.mark.parametrize("value", [-1, 0, 1, 2000])
    def test_leanvec_dims_in_range(self, value: int) -> None:
        make_config(leanvec_dims=value)

    @pytest.mark.parametrize("value", [-2, 2001])
    def test_leanvec_dims_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError, match="leanvec_dims"):
            make_config(leanvec_dims=value)

    @pytest.mark.parametrize("value", [0, 1, 2])
    def test_compression_type_valid(self, value: int) -> None:
        make_config(compression_type=value)

    @pytest.mark.parametrize("value", [-1, 3])
    def test_compression_type_invalid(self, value: int) -> None:
        with pytest.raises(ValidationError, match="compression_type"):
            make_config(compression_type=value)

    @pytest.mark.parametrize("value", [-8, -4, 4, 8])
    def test_compression_primary_valid(self, value: int) -> None:
        make_config(compression_primary=value)

    @pytest.mark.parametrize("value", [-6, 0, 6, 16])
    def test_compression_primary_invalid(self, value: int) -> None:
        with pytest.raises(ValidationError, match="compression_primary"):
            make_config(compression_primary=value)

    def test_compression_secondary_zero_requires_lvq(self):
        with pytest.raises(ValidationError, match="compression_secondary"):
            make_config(compression_type=1, compression_secondary=0)

    def test_compression_secondary_zero_allowed_for_lvq(self):
        make_config(compression_type=2, compression_primary=8, compression_secondary=0)


# -- Config validation: the (primary, secondary) pair rule -- finding #6 -----


class TestPgSVSCompressionPair:
    @pytest.mark.parametrize(("primary", "secondary"), [(4, 4), (4, 8), (8, 8), (-8, -8), (-8, 8)])
    def test_leanvec_supported_pairs(self, primary: int, secondary: int) -> None:
        make_config(compression_type=1, compression_primary=primary, compression_secondary=secondary)

    @pytest.mark.parametrize(("primary", "secondary"), [(8, 4), (8, -4)])
    def test_leanvec_unsupported_pairs(self, primary: int, secondary: int) -> None:
        with pytest.raises(ValidationError, match="LeanVec pairing"):
            make_config(compression_type=1, compression_primary=primary, compression_secondary=secondary)

    @pytest.mark.parametrize(("primary", "secondary"), [(4, 0), (8, 0), (4, 4), (4, 8), (-4, 0), (-8, 0)])
    def test_lvq_supported_pairs(self, primary: int, secondary: int) -> None:
        make_config(compression_type=2, compression_primary=primary, compression_secondary=secondary)

    @pytest.mark.parametrize(("primary", "secondary"), [(8, 4), (8, 8), (-8, -8), (-8, 4)])
    def test_lvq_unsupported_pairs(self, primary: int, secondary: int) -> None:
        with pytest.raises(ValidationError, match="LVQ pairing"):
            make_config(compression_type=2, compression_primary=primary, compression_secondary=secondary)

    def test_pair_rule_skipped_when_compression_disabled(self):
        # compression_type=0/None: primary=8/secondary=4 would be an invalid LeanVec
        # pairing, but with compression off the pair rule shouldn't even apply.
        make_config(compression_type=None, compression_primary=8, compression_secondary=4)

    def test_ui_default_pair_valid_under_both_schemes(self):
        # The Streamlit widgets' first (default) option is (4, 8); must be valid
        # for whichever scheme the user then picks.
        make_config(compression_type=1, compression_primary=4, compression_secondary=8)
        make_config(compression_type=2, compression_primary=4, compression_secondary=8)


# -- Config behavior: metric/quantization resolution -------------------------


class TestPgSVSConfigBehavior:
    @pytest.mark.parametrize(
        ("metric_type", "quantization_type", "expected"),
        [
            (MetricType.L2, None, "vector_l2_ops"),
            (MetricType.IP, None, "vector_ip_ops"),
            (MetricType.COSINE, None, "vector_cosine_ops"),
            (MetricType.L2, "halfvec", "halfvec_l2_ops"),
            (MetricType.IP, "halfvec", "halfvec_ip_ops"),
            (MetricType.COSINE, "halfvec", "halfvec_cosine_ops"),
        ],
    )
    def test_index_param_metric_resolution(
        self,
        metric_type: MetricType,
        quantization_type: str | None,
        expected: str,
    ) -> None:
        cfg = make_config(metric_type=metric_type, quantization_type=quantization_type)
        assert cfg.index_param()["metric"] == expected

    def test_index_param_normalizes_none_quantization_to_vector(self):
        cfg = make_config(quantization_type=None, table_quantization_type=None)
        params = cfg.index_param()
        assert params["quantization_type"] == "vector"
        assert params["table_quantization_type"] == "vector"

    def test_normalize_cosine_switches_effective_metric_to_l2(self):
        normalized = make_config(metric_type=MetricType.COSINE, normalize_cosine=True)
        plain = make_config(metric_type=MetricType.COSINE, normalize_cosine=False)
        assert normalized.index_param()["metric"] == "vector_l2_ops"
        assert plain.index_param()["metric"] == "vector_cosine_ops"

    def test_normalize_cosine_is_a_no_op_for_other_metrics(self):
        cfg = make_config(metric_type=MetricType.L2, normalize_cosine=True)
        assert cfg.index_param()["metric"] == "vector_l2_ops"

    def test_search_param_has_no_reranking_or_search_history_fields(self):
        # Regression guard: reranking/quantized_fetch_limit/reranking_metric were
        # removed as always-dead for svs, and use_search_history is build-time only.
        cfg = make_config(search_window_size=200, use_search_history=True)
        assert cfg.search_param() == {"metric_fun_op": "<=>", "search_window_size": 200}

    def test_session_param_emits_search_window_size(self):
        cfg = make_config(search_window_size=150)
        opts = cfg.session_param()["session_options"]
        assert opts == [{"parameter": {"setting_name": "svs.search_window_size", "val": "150"}}]

    def test_session_param_empty_when_search_window_size_unset(self):
        cfg = make_config(search_window_size=None)
        assert cfg.session_param()["session_options"] == []

    def test_index_param_includes_build_time_options(self):
        cfg = make_config(graph_degree=32, alpha=-1, compression_type=1, compression_primary=4, compression_secondary=8)
        names = {opt["option_name"]: opt["val"] for opt in cfg.index_param()["index_creation_with_options"]}
        assert names["graph_degree"] == "32"
        assert names["alpha"] == "-1"
        assert names["compression_type"] == "1"
        assert names["compression_primary"] == "4"
        assert names["compression_secondary"] == "8"
        # search_window_size is deliberately not a build-time option (session-only)
        assert "search_window_size" not in names


# -- Client: password logging -- finding #1 ----------------------------------


class TestPgSVSClientLogging:
    def test_connection_password_is_redacted(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        with capture_logger("vectordb_bench.backend.clients.pgsvs.pgsvs", caplog):
            make_db("test_redact", drop_old=False)

        assert DB_CONFIG["connect_config"]["password"] not in caplog.text
        assert "localhost" in caplog.text  # non-sensitive fields still get logged


# -- Client: index naming -- finding #2 ---------------------------------------


class TestPgSVSClientIndexName:
    @pytest.mark.parametrize("table_name", ["table_a", "table_b"])
    def test_index_name_is_table_scoped(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        table_name: str,
    ) -> None:
        db = make_db(table_name, drop_old=False)
        assert db._index_name == f"{table_name}_svs_index"

    def test_two_tables_get_distinct_index_names(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db_a = make_db("table_a", drop_old=False)
        db_b = make_db("table_b", drop_old=False)
        assert db_a._index_name != db_b._index_name

    def test_drop_old_does_not_issue_a_separate_drop_index(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_sql_rendering(monkeypatch)
        _conn, cursor = mocked_db_connection

        make_db("test_drop_old", drop_old=True)

        statements = rendered_executed_sql(cursor)
        assert not any("DROP INDEX" in s for s in statements)
        assert any("DROP TABLE" in s for s in statements)


# -- Client: search_window_size vs k -- finding #3 ----------------------------


class TestPgSVSClientSearchWindow:
    def _db_for_search(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        search_window_size: int | None,
    ) -> PgSVS:
        db = make_db(
            "test_search_window",
            drop_old=False,
            case_config=make_config(search_window_size=search_window_size),
        )
        db.conn, db.cursor = mocked_db_connection
        db._search = "SELECT 1"
        db.cursor.execute.return_value.fetchall.return_value = []
        return db

    def test_rejects_k_above_explicit_window(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = self._db_for_search(mocked_db_connection, search_window_size=100)
        with pytest.raises(ValueError, match="exceeds search_window_size"):
            db.search_embedding(query=[0.1] * DIM, k=200)

    def test_allows_k_at_explicit_window_boundary(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = self._db_for_search(mocked_db_connection, search_window_size=200)
        db.search_embedding(query=[0.1] * DIM, k=200)  # must not raise

    def test_allows_k_below_explicit_window(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = self._db_for_search(mocked_db_connection, search_window_size=300)
        db.search_embedding(query=[0.1] * DIM, k=200)  # must not raise

    def test_never_rejects_when_window_unset(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = self._db_for_search(mocked_db_connection, search_window_size=None)
        db.search_embedding(query=[0.1] * DIM, k=10_000)  # must not raise; left to svs's own default


# -- Client: reused-index validation -- finding #4 ----------------------------


class TestPgSVSClientIndexReuseValidation:
    def test_skips_when_no_index_exists_yet(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        make_db("fresh_table", drop_old=False)  # default side_effect -> fetchone() is None; must not raise

    def test_passes_when_existing_index_matches(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        _conn, cursor = mocked_db_connection
        cursor.execute.side_effect = existing_index_side_effect(
            ("vamana", "vector_l2_ops", "vector(8)", ["graph_degree=32"]),
        )
        make_db(
            "matching_table",
            drop_old=False,
            case_config=make_config(metric_type=MetricType.L2, graph_degree=32),
        )  # must not raise

    def test_rejects_wrong_access_method(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        _conn, cursor = mocked_db_connection
        cursor.execute.side_effect = existing_index_side_effect(
            ("hnsw", "vector_l2_ops", "vector(8)", []),
        )
        with pytest.raises(RuntimeError, match="access_method=hnsw"):
            make_db("wrong_am_table", drop_old=False, case_config=make_config(metric_type=MetricType.L2))

    def test_rejects_wrong_operator_class(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        _conn, cursor = mocked_db_connection
        cursor.execute.side_effect = existing_index_side_effect(
            ("vamana", "vector_cosine_ops", "vector(8)", []),
        )
        with pytest.raises(RuntimeError, match="operator_class=vector_cosine_ops"):
            make_db("wrong_opc_table", drop_old=False, case_config=make_config(metric_type=MetricType.L2))

    def test_rejects_wrong_column_type(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        _conn, cursor = mocked_db_connection
        cursor.execute.side_effect = existing_index_side_effect(
            ("vamana", "vector_l2_ops", "halfvec(8)", []),
        )
        with pytest.raises(RuntimeError, match="column_type=halfvec"):
            make_db(
                "wrong_coltype_table",
                drop_old=False,
                case_config=make_config(metric_type=MetricType.L2, table_quantization_type=None),
            )

    def test_rejects_mismatched_reloption(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        _conn, cursor = mocked_db_connection
        cursor.execute.side_effect = existing_index_side_effect(
            ("vamana", "vector_l2_ops", "vector(8)", ["graph_degree=16"]),
        )
        with pytest.raises(RuntimeError, match="graph_degree=16"):
            make_db(
                "mismatched_reloption_table",
                drop_old=False,
                case_config=make_config(metric_type=MetricType.L2, graph_degree=64),
            )


# -- Client: index-build settings stay session-scoped -- finding #5 ----------


class TestPgSVSClientBuildSettings:
    def test_build_settings_do_not_alter_the_role(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ):
        patch_sql_rendering(monkeypatch)
        conn, cursor = mocked_db_connection
        cursor.execute.return_value.fetchall.return_value = []

        db = make_db(
            "test_build_settings",
            drop_old=False,
            case_config=make_config(maintenance_work_mem="512MB", max_parallel_workers=4),
        )
        db.conn, db.cursor = conn, cursor
        db._set_parallel_index_build_param()

        statements = rendered_executed_sql(cursor)
        assert not any("ALTER USER" in s for s in statements)
        assert any("SET maintenance_work_mem" in s for s in statements)
        assert any("SET max_parallel_workers" in s for s in statements)
        # the per-table storage parameter is legitimate and should remain
        assert any("ALTER TABLE" in s and "parallel_workers" in s for s in statements)


# -- Client: general lifecycle coverage ---------------------------------------


class TestPgSVSClientLifecycle:
    def test_db_is_not_thread_safe(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = make_db("test_thread_safe", drop_old=False)
        assert db.thread_safe is False

    def test_copy_for_thread_drops_the_live_connection(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        class NonDeepcopyableHandle:
            def __deepcopy__(self, _memo: object) -> None:
                raise TypeError("live handle cannot be deep-copied")

        db = make_db("test_copy_for_thread", drop_old=False)
        live_conn = NonDeepcopyableHandle()
        live_cursor = NonDeepcopyableHandle()
        db.conn = live_conn
        db.cursor = live_cursor

        db_copy = db.copy_for_thread()

        assert db_copy is not db
        assert db_copy.conn is None
        assert db_copy.cursor is None
        assert db.conn is live_conn
        assert db.cursor is live_cursor
        assert db_copy.connect_config == db.connect_config

    def test_optimize_drops_then_recreates_the_index(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db = make_db("test_optimize", drop_old=False)
        lifecycle = MagicMock()
        monkeypatch.setattr(db, "_drop_index", lifecycle.drop_index)
        monkeypatch.setattr(db, "_create_index", lifecycle.create_index)

        db.optimize()

        assert lifecycle.mock_calls == [call.drop_index(), call.create_index()]

    def test_optimize_is_a_no_op_when_create_index_after_load_is_false(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db = make_db(
            "test_optimize_skip",
            drop_old=False,
            case_config=make_config(create_index_after_load=False, create_index_before_load=True),
        )
        lifecycle = MagicMock()
        monkeypatch.setattr(db, "_drop_index", lifecycle.drop_index)
        monkeypatch.setattr(db, "_create_index", lifecycle.create_index)

        db.optimize()

        assert lifecycle.mock_calls == []

    def test_prepare_filter_non_filter(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = make_db("test_filter_none", drop_old=False)
        db.prepare_filter(non_filter)
        assert db.where_clause == ""

    def test_prepare_filter_num_ge(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = make_db("test_filter_numge", drop_old=False)
        db.prepare_filter(IntFilter(int_value=42, filter_rate=0.5))
        assert "id >= 42" in db.where_clause

    def test_init_raises_when_neither_create_index_flag_is_set(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
    ) -> None:
        with pytest.raises(RuntimeError, match="must create an index"):
            make_db(
                "test_no_index_flags",
                drop_old=False,
                case_config=make_config(create_index_before_load=False, create_index_after_load=False),
            )
