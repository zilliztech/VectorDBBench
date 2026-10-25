"""Unit tests for the ADB-PG Nova client config layer.

These tests do not require a live database — they only exercise:
  - AdbpgConfig defaults and connection-string assembly
  - AdbpgIndexConfig.index_param() WITH-clause options (incl. raw auto_reduction)
  - AdbpgIndexConfig.session_param() fastann GUC emission
  - TestResult.read_file() round-trip when password is absent in saved JSON
    (regression for the result-loading failure caused by polymorphic
    serialization stripping subclass fields from DBConfig)

Usage:
  pytest tests/test_adbpg.py -v
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.adbpg.config import AdbpgConfig, AdbpgIndexConfig
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.models import TestResult

if TYPE_CHECKING:
    from pathlib import Path


def make_index_config(**overrides) -> AdbpgIndexConfig:
    base = {
        "metric_type": MetricType.COSINE,
        "hnsw_m": 32,
        "ef_search": 100,
        "ef_construction": 200,
        "nlist": 1024,
        "algorithm": "novamr",
        "rabitq_bits": 7,
        "quantize_rescore_amp": 1.0,
        "nova_adaptive_gamma": 0.0,
        "max_scan_points": 2000,
        "index_scan_mode": "snapshot",
        "auto_reduction": False,
        "nprobe": 5,
    }
    base.update(overrides)
    return AdbpgIndexConfig(**base)


class TestAdbpgConfig:
    def test_defaults_allow_construction_without_password(self):
        # Regression: result JSON only contains DBConfig parent fields
        # (db_label/version/note) because of pydantic polymorphic serialization.
        # AdbpgConfig must therefore be constructible from that minimal dict.
        cfg = AdbpgConfig(db_label="", version="", note="")
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.db_name == "postgres"
        assert cfg.password.get_secret_value() == ""

    def test_to_dict_carries_utility_session_option(self):
        cfg = AdbpgConfig(
            user_name=SecretStr("u"),
            password=SecretStr("pw"),
            host="h.example.com",
            port=5432,
            db_name="postgres",
        )
        d = cfg.to_dict()
        assert d["table_name"] == "vector"
        cc = d["connect_config"]
        assert cc["host"] == "h.example.com"
        assert cc["user"] == "u"
        assert cc["password"] == "pw"  # noqa: S105
        assert cc["dbname"] == "postgres"
        assert cc["options"] == "-c gp_session_role=utility"


class TestAdbpgIndexConfigBuild:
    def test_parse_metric(self):
        assert make_index_config(metric_type=MetricType.L2).parse_metric() == "l2"
        assert make_index_config(metric_type=MetricType.COSINE).parse_metric() == "cosine"
        assert make_index_config(metric_type=MetricType.IP).parse_metric() == "ip"

    def test_parse_metric_unsupported_raises(self):
        with pytest.raises(ValueError, match="Metric type"):
            make_index_config(metric_type=None).parse_metric()

    def test_index_param_options_default(self):
        params = make_index_config().index_param()
        names = {opt["option_name"]: opt for opt in params["index_creation_with_options"]}
        assert names["algorithm"]["val"] == "novamr"
        assert names["hnsw_m"]["val"] == 32
        assert names["hnsw_ef_construction"]["val"] == 200
        assert names["nlist"]["val"] == 1024
        assert names["rabitq_bits"]["val"] == 7
        assert names["max_key_len"]["val"] == 1
        # auto_reduction is omitted when False
        assert "auto_reduction" not in names

    def test_index_param_auto_reduction_emits_raw(self):
        params = make_index_config(auto_reduction=True).index_param()
        opt = next(o for o in params["index_creation_with_options"] if o["option_name"] == "auto_reduction")
        # `raw=True` so the value is rendered as a bare SQL identifier (`on`)
        # rather than a quoted literal.
        assert opt["val"] == "on"
        assert opt.get("raw") is True

    def test_index_param_pca_dim_omitted_when_none(self):
        params = make_index_config(pca_dim=None).index_param()
        names = {opt["option_name"] for opt in params["index_creation_with_options"]}
        assert "pca_dim" not in names

    def test_index_param_pca_dim_emitted_when_set(self):
        params = make_index_config(pca_dim=448).index_param()
        opt = next(o for o in params["index_creation_with_options"] if o["option_name"] == "pca_dim")
        assert opt["val"] == 448


class TestAdbpgIndexConfigSession:
    def test_session_param_emits_all_search_gucs(self):
        cfg = make_index_config(
            quantize_rescore_amp=0.6,
            nova_adaptive_gamma=0.0,
            ef_search=50,
            max_scan_points=16000,
            index_scan_mode="snapshot",
            nprobe=64,
        )
        opts = cfg.session_param()["session_options"]
        emitted = {o["parameter"]["setting_name"]: o["parameter"]["val"] for o in opts}
        assert emitted["fastann.quantize_rescore_amp"] == "0.6"
        assert emitted["fastann.nova_adaptive_gamma"] == "0.0"
        assert emitted["fastann.hnsw_ef_search"] == "50"
        assert emitted["fastann.hnsw_max_scan_points"] == "16000"
        assert emitted["fastann.index_scan_mode"] == "snapshot"
        # novad-specific GUC is always emitted (no-op for HNSW algorithms)
        assert emitted["fastann.nova_nprobe"] == "64"

    def test_session_param_emits_zero_values(self):
        # Forcing 0 / 0.0 must still produce a SET command — callers rely on
        # being able to pin a GUC to zero.
        cfg = make_index_config(quantize_rescore_amp=0.0, nova_adaptive_gamma=0.0, nprobe=0)
        opts = cfg.session_param()["session_options"]
        emitted = {o["parameter"]["setting_name"]: o["parameter"]["val"] for o in opts}
        assert emitted["fastann.quantize_rescore_amp"] == "0.0"
        assert emitted["fastann.nova_adaptive_gamma"] == "0.0"
        assert emitted["fastann.nova_nprobe"] == "0"


class TestResultRoundTrip:
    def test_read_file_with_minimal_db_config(self, tmp_path: Path):
        """Saved result JSON keeps only DBConfig parent fields for db_config.

        TestResult.read_file must still rehydrate the AdbpgConfig instance
        without raising a Field-required pydantic ValidationError.
        """
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
                        "db_case_config": {
                            "metric_type": "COSINE",
                            "algorithm": "novamr",
                            "hnsw_m": 16,
                            "ef_search": 100,
                            "ef_construction": 200,
                            "nlist": 1024,
                        },
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

        tr = TestResult.read_file(result_file, trans_unit=False)
        assert len(tr.results) == 1
        rehydrated = tr.results[0].task_config.db_config
        assert isinstance(rehydrated, AdbpgConfig)
        assert rehydrated.host == "localhost"  # came from default
