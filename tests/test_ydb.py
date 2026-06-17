from unittest.mock import MagicMock, patch
import os

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.backend.clients.ydb.config import (
    YDBConfig,
    YDBIndexConfig,
)
from vectordb_bench.backend.clients.ydb.ydb_client import YDB
from vectordb_bench.backend.filter import IntFilter, non_filter


def _integration_db_config() -> dict:
    return YDBConfig(
        endpoint=os.environ.get("YDB_ENDPOINT", "grpc://localhost:2136"),
        database=os.environ.get("YDB_DATABASE", "/Root/test"),
        auth_mode=os.environ.get("YDB_AUTH_MODE", "env"),
    ).to_dict()


class TestYDBConfig:
    def test_index_param_omits_unset_kmeans_options(self):
        cfg = YDBIndexConfig()
        params = cfg.index_param()
        assert params["levels"] is None
        assert params["clusters"] is None
        assert params["overlap_clusters"] == 3

    def test_index_param_passes_explicit_values(self):
        cfg = YDBIndexConfig(levels=2, clusters=64, overlap_clusters=5)
        params = cfg.index_param()
        assert params["levels"] == 2
        assert params["clusters"] == 64
        assert params["overlap_clusters"] == 5

    def test_metric_mapping(self):
        cosine = YDBIndexConfig(metric_type=MetricType.COSINE)
        assert cosine.index_strategy() == "similarity=cosine"
        assert cosine.knn_function() == "CosineSimilarity"

        l2 = YDBIndexConfig(metric_type=MetricType.L2)
        assert l2.index_strategy() == "distance=euclidean"
        assert l2.knn_function() == "EuclideanDistance"

        ip = YDBIndexConfig(metric_type=MetricType.IP)
        assert ip.index_strategy() == "similarity=inner_product"
        assert ip.knn_function() == "InnerProductSimilarity"

    def test_default_search_top_size(self):
        cfg = YDBIndexConfig()
        assert cfg.num_leaves_to_search == 40
        assert cfg.search_param()["kmeans_tree_search_top_size"] == 40

    def test_index_on_columns(self):
        from vectordb_bench.backend.filter import IntFilter, LabelFilter, non_filter

        cfg = YDBIndexConfig()
        assert cfg.index_on_columns(non_filter) == ("embedding",)
        assert cfg.index_on_columns(non_filter, with_scalar_labels=True) == ("labels", "embedding")
        assert cfg.index_on_columns(IntFilter(int_value=100, filter_rate=0.01)) == ("id", "embedding")
        assert cfg.index_on_columns(LabelFilter(label_percentage=0.01)) == ("labels", "embedding")

    def test_cover_clause_for_label_table(self):
        assert YDBIndexConfig(cover_embedding=False).cover_clause(with_scalar_labels=True) == "COVER (embedding)"
        assert YDBIndexConfig(cover_embedding=True).cover_clause(with_scalar_labels=True) == "COVER (embedding)"
        assert YDBIndexConfig(cover_embedding=False).cover_clause() == ""

    def test_cover_clause(self):
        assert YDBIndexConfig(cover_embedding=True).cover_clause() == "COVER (embedding)"
        assert YDBIndexConfig(cover_embedding=False).cover_clause() == ""

    def test_zero_means_auto_for_index_shape(self):
        cfg = YDBIndexConfig(level=0, nlist=0, overlap_clusters=0)
        assert cfg.level is None
        assert cfg.nlist is None
        assert cfg.overlap_clusters is None

    def test_auto_partitioning_defaults(self):
        cfg = YDBConfig()
        assert cfg.auto_partitioning_min_partitions_count == 1000
        assert cfg.auto_partitioning_max_partitions_count == 1100
        assert cfg.auto_partitioning_table_partition_size_mb == 1000
        assert cfg.auto_partitioning_index_partition_size_mb == 1000
        assert cfg.table_name == ""
        assert cfg.operation_timeout_seconds == 24 * 3600

    def test_operation_settings_timeout(self):
        settings = YDB._operation_settings({"operation_timeout_seconds": 7200})
        assert settings.timeout == 7200
        assert settings.operation_timeout == 7200
        assert settings.cancel_after == 7200

    def test_empty_table_name_allowed(self):
        cfg = YDBConfig(table_name="")
        assert cfg.table_name == ""

    def test_table_name_from_db_config_overrides_collection_name(self):
        client = YDB(
            dim=4,
            db_config=YDBConfig(table_name="explicit_table").to_dict(),
            db_case_config=YDBIndexConfig(),
            collection_name="generated_name",
            drop_old=False,
        )
        assert client.table_name == "explicit_table"

    def test_collection_name_used_when_table_name_unset(self):
        client = YDB(
            dim=4,
            db_config=YDBConfig().to_dict(),
            db_case_config=YDBIndexConfig(),
            collection_name="generated_name",
            drop_old=False,
        )
        assert client.table_name == "generated_name"

    def test_runner_capability_flags(self):
        assert YDB.serial_search_in_process is True
        assert YDB.case_unique_collection_name is True
        assert YDB.case_filters_at_init is True
        assert YDB.optimize_via_picklable_worker is True

    def test_auto_partitioning_bounds_validation(self):
        with pytest.raises(ValueError, match="auto_partitioning_min_partitions_count"):
            YDBConfig(
                auto_partitioning_min_partitions_count=1200,
                auto_partitioning_max_partitions_count=1100,
            )


class TestYDBTableDDL:
    def _make_client(self) -> YDB:
        client = YDB.__new__(YDB)
        client.db_config = YDBConfig(
            auto_partitioning_min_partitions_count=1000,
            auto_partitioning_max_partitions_count=1100,
        ).to_dict()
        client.table_name = "bench_table"
        client.index_name = "vector_idx"
        client.filters = non_filter
        client.with_scalar_labels = False
        return client

    def test_create_table_with_auto_partitioning(self):
        client = self._make_client()
        ddl = client._create_table_with_clause()
        assert "AUTO_PARTITIONING_BY_SIZE = ENABLED" in ddl
        assert "AUTO_PARTITIONING_BY_LOAD = ENABLED" in ddl
        assert "AUTO_PARTITIONING_PARTITION_SIZE_MB = 1000" in ddl
        assert "AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = 1000" in ddl
        assert "AUTO_PARTITIONING_MAX_PARTITIONS_COUNT = 1100" in ddl

    def test_index_impl_tables_global_cover(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(cover_embedding=True)
        client.filters = non_filter
        assert client._index_impl_table_paths() == [
            "bench_table/vector_idx/indexImplLevelTable",
            "bench_table/vector_idx/indexImplPostingTable",
        ]

    def test_index_impl_tables_without_cover(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(cover_embedding=False)
        assert client._index_impl_table_paths() == [
            "bench_table/vector_idx/indexImplLevelTable",
        ]

    def test_index_impl_tables_with_filter_prefix(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(cover_embedding=True)
        client.filters = IntFilter(int_value=100, filter_rate=0.01)
        assert client._index_impl_table_paths() == [
            "bench_table/vector_idx/indexImplLevelTable",
            "bench_table/vector_idx/indexImplPostingTable",
            "bench_table/vector_idx/indexImplPrefixTable",
        ]

    def test_index_on_sql_for_label_table(self):
        from vectordb_bench.backend.filter import LabelFilter

        client = self._make_client()
        client.with_scalar_labels = True
        client.filters = LabelFilter(label_percentage=0.01)
        client.case_config = YDBIndexConfig(cover_embedding=True)
        assert client._index_on_sql() == "labels, embedding"
        index_param = client.case_config.index_param(
            client.filters,
            with_scalar_labels=True,
        )
        assert index_param["on_columns"] == ("labels", "embedding")
        assert index_param["cover_clause"] == "COVER (embedding)"

    def test_index_impl_tables_with_label_filter(self):
        from vectordb_bench.backend.filter import LabelFilter

        client = self._make_client()
        client.with_scalar_labels = True
        client.filters = LabelFilter(label_percentage=0.01)
        client.case_config = YDBIndexConfig(cover_embedding=True)
        assert client._index_impl_table_paths() == [
            "bench_table/vector_idx/indexImplLevelTable",
            "bench_table/vector_idx/indexImplPostingTable",
            "bench_table/vector_idx/indexImplPrefixTable",
        ]

    def test_configure_index_table_partitioning_sql(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(cover_embedding=True)
        settings = client._index_partitioning_settings_sql()
        assert "AUTO_PARTITIONING_PARTITION_SIZE_MB = 1000" in settings
        assert "AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = 1000" in settings

    def _capture_add_index_sql(self, client: YDB) -> str:
        import sys

        captured: dict[str, str] = {}

        class _Pool:
            def execute_with_retries(self, query, *args, **kwargs):
                if "ADD INDEX" in query:
                    captured["query"] = query

        client.dim = 4
        client.driver = MagicMock()
        mock_ydb = MagicMock()
        mock_ydb.RetrySettings = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {"ydb": mock_ydb}), patch.object(
            YDB, "_operation_settings", return_value=MagicMock()
        ), patch.object(YDB, "_try_index_search", return_value=False):
            client._add_vector_index(_Pool())
        return captured["query"]

    def test_add_vector_index_uses_fixed_index_name(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        sql = self._capture_add_index_sql(client)
        assert "ADD INDEX vector_idx" in sql
        assert "__temp" not in sql

    def test_add_vector_index_treats_path_exist_as_build_in_progress(self):
        client = self._make_client()
        client.dim = 4
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        attempts = {"count": 0}

        class _Pool:
            def execute_with_retries(self, query, *args, **kwargs):
                if "ADD INDEX" in query:
                    attempts["count"] += 1
                    raise RuntimeError("path exist for vector_idx")

        with patch.object(YDB, "_operation_settings", return_value=MagicMock()), patch.object(
            YDB, "_try_index_search", return_value=False
        ), patch.object(YDB, "_drop_vector_indexes"):
            client._add_vector_index(_Pool())

        assert attempts["count"] == 1

    def test_add_vector_index_treats_transient_error_as_build_in_progress(self):
        client = self._make_client()
        client.dim = 4
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        attempts = {"count": 0}

        class _Pool:
            def execute_with_retries(self, query, *args, **kwargs):
                if "ADD INDEX" in query:
                    attempts["count"] += 1
                    raise RuntimeError(
                        'Unavailable: message: "Connection to tablet was lost." severity: 1 (server_code: 400050)'
                    )

        with patch.object(YDB, "_operation_settings", return_value=MagicMock()), patch.object(
            YDB, "_try_index_search", return_value=False
        ), patch.object(YDB, "_drop_vector_indexes"), patch.object(YDB, "_reconnect") as mock_reconnect:
            client._add_vector_index(_Pool())

        assert attempts["count"] == 1
        mock_reconnect.assert_called_once()

    def test_is_transient_ydb_error_detects_tablet_connection_loss(self):
        exc = RuntimeError('Unavailable: message: "Connection to tablet was lost." severity: 1 (server_code: 400050)')
        assert YDB._is_transient_ydb_error(exc) is True
        assert YDB._is_index_build_in_progress_error(exc) is True

    def test_wait_for_index_reconnects_on_transient_probe_errors(self):
        client = self._make_client()
        client.dim = 4
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        client._index_ready = False
        statuses = iter(["connection_error", "ready"])

        with patch.object(YDB, "_probe_index_status", side_effect=lambda pool: next(statuses)), patch.object(
            YDB, "_reconnect"
        ) as mock_reconnect, patch("vectordb_bench.backend.clients.ydb.ydb_client.time.sleep"):
            client._wait_for_index(MagicMock())

        assert client._index_ready is True
        mock_reconnect.assert_called_once()

    def test_add_vector_index_skips_create_when_already_searchable(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        attempts = {"count": 0}

        class _Pool:
            def execute_with_retries(self, query, *args, **kwargs):
                attempts["count"] += 1

        with patch.object(YDB, "_try_index_search", return_value=True):
            client._add_vector_index(_Pool())

        assert attempts["count"] == 0
        assert client._index_ready is True

    def test_ddl_retry_settings_disable_internal_retries(self):
        client = self._make_client()
        settings = client._ddl_retry_settings()
        assert settings.max_retries == 0

    def test_driver_wait_timeout_is_extended(self):
        from vectordb_bench.backend.clients.ydb.ydb_client import YDB_DRIVER_WAIT_SECONDS

        assert YDB_DRIVER_WAIT_SECONDS >= 30

    def test_drop_vector_indexes_issues_drop_for_final_and_legacy_temp(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        queries: list[str] = []

        class _Pool:
            def execute_with_retries(self, query, *args, **kwargs):
                queries.append(query)

        with patch.object(YDB, "_operation_settings", return_value=MagicMock()):
            client._drop_vector_indexes(_Pool())
        assert len(queries) == 4
        assert "DROP INDEX `vector_idx`" in queries[0]
        assert "DROP INDEX `vector_idx__temp`" in queries[1]
        assert "DROP INDEX `bench_table_vector_idx`" in queries[2]
        assert "DROP INDEX `bench_table_vector_idx__temp`" in queries[3]

    def test_add_vector_index_omits_unset_kmeans_options(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE)
        sql = self._capture_add_index_sql(client)
        assert "levels=" not in sql
        assert "\n                clusters=" not in sql
        assert "overlap_clusters=3" in sql
        assert "vector_dimension=4" in sql

    def test_add_vector_index_includes_explicit_kmeans_options(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(
            metric_type=MetricType.COSINE,
            levels=2,
            clusters=64,
            overlap_clusters=5,
        )
        sql = self._capture_add_index_sql(client)
        assert "levels=2" in sql
        assert "clusters=64" in sql
        assert "overlap_clusters=5" in sql

    def test_add_vector_index_partial_kmeans_options(self):
        client = self._make_client()
        client.case_config = YDBIndexConfig(metric_type=MetricType.COSINE, levels=3)
        sql = self._capture_add_index_sql(client)
        assert "levels=3" in sql
        assert "\n                clusters=" not in sql
        assert "overlap_clusters=3" in sql


class TestYDBUIConfig:
    def test_ydb_is_in_ui_db_list(self):
        from vectordb_bench.frontend.config.dbCaseConfigs import DB_LIST, CASE_CONFIG_MAP

        assert DB.YDB in DB_LIST
        assert DB.YDB in CASE_CONFIG_MAP

    def test_ui_case_config_maps_to_ydb_index_config(self):
        from vectordb_bench.backend.cases import CaseLabel
        from vectordb_bench.frontend.config.dbCaseConfigs import get_case_config_inputs

        ui_cfg = {
            c.label.value: c.inputConfig["value"]
            for c in get_case_config_inputs(DB.YDB, CaseLabel.Performance)
        }
        inst = DB.YDB.case_config_cls(None)(**ui_cfg)
        assert inst.overlap_clusters == 3
        assert inst.num_leaves_to_search == 40
        assert inst.level is None
        assert inst.nlist is None


class TestYDBAuth:
    def test_build_credentials_login_from_env(self, monkeypatch):
        monkeypatch.setenv("YDB_USER", "bench")
        monkeypatch.setenv("YDB_PASSWORD", "secret")
        monkeypatch.delenv("YDB_SSL_ROOT_CERTIFICATES_FILE", raising=False)

        with patch("ydb.StaticCredentials") as static_credentials, patch("ydb.DriverConfig") as driver_config_cls:
            YDB._build_credentials(
                {"auth_mode": "env", "user": "", "password": "", "endpoint": "grpc://localhost:2136", "database": "/local"}
            )
            driver_config_cls.assert_called_once_with(endpoint="grpc://localhost:2136", database="/local")
            static_credentials.assert_called_once_with(driver_config_cls.return_value, "bench", "secret")

    def test_build_credentials_login_mode_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("YDB_USER", "env-user")
        monkeypatch.setenv("YDB_PASSWORD", "env-pass")
        monkeypatch.delenv("YDB_SSL_ROOT_CERTIFICATES_FILE", raising=False)

        with patch("ydb.StaticCredentials") as static_credentials, patch("ydb.DriverConfig") as driver_config_cls:
            YDB._build_credentials(
                {
                    "auth_mode": "login",
                    "user": "cli-user",
                    "password": "cli-pass",
                    "endpoint": "grpc://localhost:2136",
                    "database": "/Root",
                }
            )
            driver_config_cls.assert_called_once_with(endpoint="grpc://localhost:2136", database="/Root")
            static_credentials.assert_called_once_with(driver_config_cls.return_value, "cli-user", "cli-pass")

    def test_build_credentials_login_mode_requires_user(self):
        with pytest.raises(ValueError, match="YDB_USER"):
            YDB._build_credentials({"auth_mode": "login", "user": "", "password": ""})

    def test_build_credentials_anonymous(self):
        with patch("ydb.AnonymousCredentials") as anonymous_credentials:
            YDB._build_credentials({"auth_mode": "anonymous"})
            anonymous_credentials.assert_called_once_with()

    def test_build_credentials_env_defaults_to_anonymous(self, monkeypatch):
        monkeypatch.delenv("YDB_USER", raising=False)
        monkeypatch.delenv("YDB_PASSWORD", raising=False)
        monkeypatch.delenv("YDB_ACCESS_TOKEN_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_OAUTH2_KEY_FILE", raising=False)
        monkeypatch.delenv("YDB_ANONYMOUS_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_METADATA_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_SSL_ROOT_CERTIFICATES_FILE", raising=False)

        with patch("ydb.AnonymousCredentials") as anonymous_credentials:
            YDB._build_credentials({"auth_mode": "env", "user": "", "password": ""})
            anonymous_credentials.assert_called_once_with()


class TestYDBSSL:
    def test_ssl_root_certificates_from_env(self, monkeypatch, tmp_path):
        cert_file = tmp_path / "ca.pem"
        cert_file.write_bytes(b"-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n")
        monkeypatch.setenv("YDB_SSL_ROOT_CERTIFICATES_FILE", str(cert_file))

        db_config = {
            "endpoint": "grpcs://ydb.example.com:2135",
            "database": "/Root/db",
        }
        loaded = YDB._load_root_certificates(db_config)
        assert loaded == cert_file.read_bytes()

    def test_ssl_root_certificates_from_db_config(self, tmp_path):
        cert_file = tmp_path / "ca.pem"
        cert_file.write_bytes(b"pem-bytes")
        db_config = {
            "endpoint": "grpcs://ydb.example.com:2135",
            "database": "/Root/db",
            "ssl_root_certificates_file": str(cert_file),
        }
        assert YDB._load_root_certificates(db_config) == b"pem-bytes"

    def test_driver_config_includes_root_certificates(self, tmp_path):
        cert_file = tmp_path / "ca.pem"
        cert_file.write_bytes(b"pem-bytes")
        db_config = {
            "endpoint": "grpcs://ydb.example.com:2135",
            "database": "/Root/db",
            "ssl_root_certificates_file": str(cert_file),
        }
        with patch("ydb.DriverConfig") as driver_config_cls:
            YDB._driver_config(db_config)
            driver_config_cls.assert_called_once_with(
                endpoint="grpcs://ydb.example.com:2135",
                database="/Root/db",
                root_certificates=b"pem-bytes",
            )

    def test_ydb_config_reads_ssl_env(self, monkeypatch, tmp_path):
        cert_file = tmp_path / "ca.pem"
        cert_file.write_text("dummy")
        monkeypatch.setenv("YDB_SSL_ROOT_CERTIFICATES_FILE", str(cert_file))
        cfg = YDBConfig()
        assert cfg.ssl_root_certificates_file == str(cert_file)

    def test_build_credentials_env_uses_sdk_when_configured(self, monkeypatch):
        monkeypatch.delenv("YDB_USER", raising=False)
        monkeypatch.setenv("YDB_ACCESS_TOKEN_CREDENTIALS", "token-value")

        with patch("ydb.credentials_from_env_variables") as from_env:
            from_env.return_value = MagicMock()
            YDB._build_credentials({"auth_mode": "env", "user": "", "password": ""})
            from_env.assert_called_once_with()


class TestYDBFilters:
    def test_prepare_filter_int(self):
        client = YDB.__new__(YDB)
        client._label_filter_value = None
        client.prepare_filter(IntFilter(int_value=12345, filter_rate=0.01))
        assert client._where_clause == "WHERE id >= 12345"
        assert client._label_filter_value is None

    def test_prepare_filter_label_uses_parameter(self):
        from vectordb_bench.backend.filter import LabelFilter

        client = YDB.__new__(YDB)
        client._label_filter_value = None
        client.prepare_filter(LabelFilter(label_percentage=0.05))
        assert client._where_clause == "WHERE labels = $label"
        assert client._label_filter_value == "label_5p"

    def test_supported_filter_types(self):
        assert FilterOp.NumGE in YDB.supported_filter_types
        assert FilterOp.StrEqual in YDB.supported_filter_types


class TestYDBPickle:
    def _make_client(self) -> YDB:
        client = YDB.__new__(YDB)
        client.name = "YDB"
        client.db_config = YDBConfig().to_dict()
        client.case_config = YDBIndexConfig()
        client.table_name = "bench_table"
        client.index_name = "vector_idx"
        client.dim = 768
        client.filters = non_filter
        client.with_scalar_labels = False
        client._where_clause = ""
        client._label_filter_value = None
        client._index_ready = False
        client.driver = object()
        client.pool = object()
        return client

    def test_db_picklable_without_live_session(self):
        import pickle

        client = self._make_client()
        restored = pickle.loads(pickle.dumps(client))  # noqa: S301
        assert restored.driver is None
        assert restored.pool is None
        assert restored.dim == 768
        assert restored.table_name == "bench_table"

    def test_getstate_strips_live_session(self):
        client = self._make_client()
        state = client.__getstate__()
        assert state["driver"] is None
        assert state["pool"] is None


@pytest.mark.integration
class TestYDBClient:
    @pytest.fixture
    def db_client(self):
        db_cls = DB.YDB.init_cls
        db_config = _integration_db_config()

        dim = 16
        try:
            client = db_cls(
                dim=dim,
                db_config=db_config,
                db_case_config=YDBIndexConfig(
                    metric_type=MetricType.COSINE,
                    levels=1,
                    clusters=8,
                ),
                collection_name="vdbbench_ydb_test",
                drop_old=True,
            )
        except (TimeoutError, OSError) as e:
            pytest.skip(f"YDB is not available at {db_config['endpoint']}{db_config['database']}: {e}")
        except Exception as e:
            if e.__class__.__module__.startswith("ydb"):
                pytest.skip(f"YDB is not available at {db_config['endpoint']}{db_config['database']}: {e}")
            raise
        return client, dim

    def test_insert_optimize_and_search(self, db_client):
        client, dim = db_client
        count = 1000
        embeddings = np.random.default_rng(42).random((count, dim)).tolist()

        with client.init():
            inserted, err = client.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert err is None
            assert inserted == count

            client.optimize(data_size=count)

            test_id = 42
            results = client.search_embedding(query=embeddings[test_id], k=10)
            assert len(results) == 10
            assert int(results[0]) == test_id
