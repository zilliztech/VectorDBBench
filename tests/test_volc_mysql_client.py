from vectordb_bench.backend.clients.volc_mysql.volc_mysql import VolcMySQL


def test_volc_mysql_is_not_thread_safe():
    # mysql.connector is not thread-safe; ConcurrentInsertRunner relies on
    # this class attribute to clamp max_workers=1.
    assert VolcMySQL.thread_safe is False


def test_volc_mysql_ready_to_load_does_not_return_falsy_none():
    # The base VectorDB has no ready_to_load() method to override here;
    # the previous stub returned None, which any `if not db.ready_to_load()`
    # caller would misread as "not ready".
    assert not hasattr(VolcMySQL, "ready_to_load") or VolcMySQL.ready_to_load.__qualname__.startswith("VectorDB")


from vectordb_bench.backend.filter import FilterOp


def test_volc_mysql_supports_numge_filter():
    # Matches the WHERE id >= %s prepared SQL the client already builds in init().
    assert FilterOp.NumGE in VolcMySQL.supported_filter_types
    assert FilterOp.NonFilter in VolcMySQL.supported_filter_types


from pydantic import SecretStr

from vectordb_bench.backend.clients.volc_mysql.config import VolcMySQLConfig


def test_volc_mysql_config_to_dict_has_only_driver_kwargs():
    cfg = VolcMySQLConfig(password=SecretStr("pw"))
    d = cfg.to_dict()
    # mysql.connector.connect accepts these; the previously-declared
    # read_timeout/write_timeout are not valid kwargs for that driver.
    assert set(d.keys()) == {"host", "port", "user", "password"}
    assert d["user"] == "root"
    assert d["password"] == "pw"


from vectordb_bench.backend.clients.volc_mysql.config import VolcMySQLHNSWConfig


def test_volc_mysql_hnsw_config_optional_fields_default_to_none():
    cfg = VolcMySQLHNSWConfig()
    assert cfg.M is None
    assert cfg.ef_search is None
    assert cfg.ef_construction is None
    assert cfg.quant_algorithm is None
    assert cfg.quant_type is None


from vectordb_bench.backend.clients.volc_mysql.volc_mysql import _build_index_attrs_json


def test_build_index_attrs_json_drops_none_values():
    j = _build_index_attrs_json({
        "metric_type": "l2",
        "M": 16,
        "ef_construction": 128,
        "quant_algorithm": None,
        "quant_type": None,
    })
    # Compact JSON, deterministic key order (insertion order from helper).
    assert j == '{"algorithm":"hnsw","distance":"l2","m":16,"ef_construction":128}'


def test_build_index_attrs_json_includes_quantization_when_set():
    j = _build_index_attrs_json({
        "metric_type": "cosine",
        "M": 32,
        "ef_construction": 200,
        "quant_algorithm": "SQ",
        "quant_type": "8_bit",
    })
    assert j == (
        '{"algorithm":"hnsw","distance":"cosine","m":32,"ef_construction":200,'
        '"quant_algorithm":"SQ","quant_type":"8_bit"}'
    )


from unittest.mock import MagicMock

from pydantic import SecretStr as _SecretStr

from vectordb_bench.backend.filter import IntFilter, NonFilter


def _make_client_with_mock_cursor():
    """Build a VolcMySQL without touching a DB (drop_old=False does not connect),
    then wire up the prebuilt SQL templates and a mock cursor so search_embedding
    can be exercised offline.
    """
    cfg = VolcMySQLConfig(password=_SecretStr("pw")).to_dict()
    client = VolcMySQL(dim=4, db_config=cfg, db_case_config=VolcMySQLHNSWConfig(), drop_old=False)
    client.select_sql = "SELECT id FROM t ORDER BY d LIMIT %s"
    client.select_sql_with_filter = "SELECT id FROM t WHERE id >= %s ORDER BY d LIMIT %s"
    client._binary_vec = False
    cursor = MagicMock()
    cursor.fetchall.return_value = [(1,), (2,)]
    client.conn = MagicMock()
    client.cursor = cursor
    return client, cursor


def test_search_embedding_uses_unfiltered_sql_by_default():
    client, cursor = _make_client_with_mock_cursor()
    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    sql, params = cursor.execute.call_args[0]
    assert sql == client.select_sql
    assert params == ("[0.1, 0.2, 0.3, 0.4]", 10)


def test_search_embedding_applies_numge_filter_after_prepare_filter():
    # Regression guard: runners apply filters via prepare_filter(), never by
    # passing filters to search_embedding. Without prepare_filter wiring the
    # client silently ran an unfiltered search and reported wrong recall.
    client, cursor = _make_client_with_mock_cursor()
    client.prepare_filter(IntFilter(int_value=500, filter_rate=0.99))
    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    sql, params = cursor.execute.call_args[0]
    assert sql == client.select_sql_with_filter
    assert params == (500, "[0.1, 0.2, 0.3, 0.4]", 10)


def test_prepare_filter_nonfilter_resets_to_unfiltered():
    client, cursor = _make_client_with_mock_cursor()
    client.prepare_filter(IntFilter(int_value=500, filter_rate=0.99))
    client.prepare_filter(NonFilter())
    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    sql, _ = cursor.execute.call_args[0]
    assert sql == client.select_sql


import tempfile

import pytest

from vectordb_bench.backend.clients.volc_mysql import volc_mysql as volc_mysql_module


def _make_uninitialized_client():
    cfg = VolcMySQLConfig(password=_SecretStr("pw")).to_dict()
    return VolcMySQL(dim=4, db_config=cfg, db_case_config=VolcMySQLHNSWConfig(), drop_old=False)


def test_init_closes_connection_when_create_database_fails(monkeypatch):
    """If admin_cursor.execute(CREATE DATABASE ...) raises inside init(),
    the open connection and both cursors must still be closed and the
    instance attributes reset to None. Prior to the fix this leaked because
    setup ran outside the try/finally.
    """
    client = _make_uninitialized_client()
    conn = MagicMock(name="conn")
    cursor = MagicMock(name="cursor")
    admin_cursor = MagicMock(name="admin_cursor")
    admin_cursor.execute.side_effect = RuntimeError("CREATE DATABASE failed")
    monkeypatch.setattr(client, "_create_connection", lambda: (conn, cursor, admin_cursor))

    with pytest.raises(RuntimeError, match="CREATE DATABASE failed"):
        with client.init():
            pass

    cursor.close.assert_called_once()
    admin_cursor.close.assert_called_once()
    conn.close.assert_called_once()
    assert client.conn is None
    assert client.cursor is None
    assert client.admin_cursor is None


def test_select_sql_does_not_hardcode_force_index(monkeypatch):
    """Streaming/read_write cases search before optimize() creates idx_v,
    so FORCE INDEX(idx_v) in the prebuilt SQL would fail every pre-optimize
    search with "Key 'idx_v' doesn't exist in table".
    """
    from vectordb_bench.backend.clients.api import MetricType

    cfg = VolcMySQLConfig(password=_SecretStr("pw")).to_dict()
    case_cfg = VolcMySQLHNSWConfig(metric_type=MetricType.COSINE)
    client = VolcMySQL(dim=4, db_config=cfg, db_case_config=case_cfg, drop_old=False)
    conn = MagicMock()
    cursor = MagicMock()
    admin_cursor = MagicMock()
    monkeypatch.setattr(client, "_create_connection", lambda: (conn, cursor, admin_cursor))
    monkeypatch.setattr(client, "_probe_binary_support", lambda: False)

    with client.init():
        assert "FORCE INDEX" not in client.select_sql
        assert "FORCE INDEX" not in client.select_sql_with_filter
        assert "idx_v" not in client.select_sql
        assert "idx_v" not in client.select_sql_with_filter


def test_insert_embeddings_returns_actual_rowcount_on_partial_load(tmp_path, monkeypatch):
    """LOAD DATA is already committed when the rowcount mismatch is detected;
    reporting (0, err) corrupts the serial runner's already_insert_count and
    sends its retry into a PK-collision wall. Return (actual, err) so the
    runner slices past the already-loaded prefix on retry.
    """
    client = _make_uninitialized_client()
    client._batch_counter = 0
    client._binary_vec = False
    cursor = MagicMock()
    cursor.rowcount = 7
    client.cursor = cursor
    client.conn = MagicMock()
    monkeypatch.setattr(volc_mysql_module.tempfile, "gettempdir", lambda: str(tmp_path))

    embeddings = [[float(i)] * 4 for i in range(10)]
    metadata = list(range(10))
    actual, err = client.insert_embeddings(embeddings, metadata)

    assert actual == 7
    assert isinstance(err, RuntimeError)
    assert "wrote 7 rows, expected 10" in str(err)


def test_insert_embeddings_returns_n_on_full_load(tmp_path, monkeypatch):
    """Sanity: when rowcount matches, return (n, None). Guards against the
    partial-load fix regressing the happy path.
    """
    client = _make_uninitialized_client()
    client._batch_counter = 0
    client._binary_vec = False
    cursor = MagicMock()
    cursor.rowcount = 10
    client.cursor = cursor
    client.conn = MagicMock()
    monkeypatch.setattr(volc_mysql_module.tempfile, "gettempdir", lambda: str(tmp_path))

    embeddings = [[float(i)] * 4 for i in range(10)]
    metadata = list(range(10))
    actual, err = client.insert_embeddings(embeddings, metadata)
    assert actual == 10
    assert err is None
