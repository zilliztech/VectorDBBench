import pytest
from click.testing import CliRunner

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.volcano_milvus import cli as volcano_milvus_cli
from vectordb_bench.frontend.config.dbCaseConfigs import (
    VolcanoMilvusLoadConfig,
    VolcanoMilvusPerformanceConfig,
)
from vectordb_bench.models import CaseConfigParamType


def test_volcano_milvus_diskann_load_param_contains_enable_prefetch():
    """enable_prefetch is a collection-load knob (knowhere.enable_prefetch),
    it must be emitted by load_param() and must NOT leak into search_param()."""
    config_cls = DB.VolcanoMilvus.case_config_cls(index_type=IndexType.DISKANN)

    case_config = config_cls(search_list=200, enable_prefetch=True)

    assert case_config.load_param()["knowhere.enable_prefetch"] == "true"
    assert case_config.search_param()["params"] == {"search_list": 200}


def test_volcano_milvus_frontend_exposes_enable_prefetch():
    load_labels = [config.label for config in VolcanoMilvusLoadConfig]
    performance_labels = [config.label for config in VolcanoMilvusPerformanceConfig]

    assert CaseConfigParamType.enable_prefetch in load_labels
    assert CaseConfigParamType.enable_prefetch in performance_labels


def test_volcano_milvus_cli_passes_enable_prefetch(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(volcano_milvus_cli, "run", fake_run)

    volcano_milvus_cli.VolcanoMilvusDISKANN.callback(
        db_label="test-volcano",
        uri="http://localhost:19530",
        user_name=None,
        password=None,
        num_shards=1,
        replica_number=1,
        load_reqs_size=int(1.5 * 1024 * 1024),
        load_after_compaction=False,
        search_list=200,
        build_search_list=200,
        max_degree=56,
        legacy=False,
        store_strategy="MEMORY",
        quant_type="RABITQ",
        num_threads=4,
        distance_strategy="QUANT THEN MORE BITS",
        enable_prefetch=True,
        enable_thp=False,
    )

    assert captured["db_case_config"].enable_prefetch is True


def test_volcano_milvus_click_flag_parses_enable_prefetch(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(volcano_milvus_cli, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        volcano_milvus_cli.VolcanoMilvusDISKANN,
        ["--uri", "http://localhost:19530", "--enable-prefetch"],
    )

    assert result.exit_code == 0, result.output
    assert captured["db_case_config"].enable_prefetch is True


def test_volcano_milvus_load_config_no_search_list():
    """search_list is a query-time param, must not appear in load config."""
    load_labels = [config.label for config in VolcanoMilvusLoadConfig]
    assert CaseConfigParamType.SearchList not in load_labels


def test_volcano_milvus_performance_has_build_search_list():
    """Performance/Streaming case also triggers index build, build_search_list
    must be tunable there as well."""
    performance_labels = [config.label for config in VolcanoMilvusPerformanceConfig]
    assert CaseConfigParamType.BuildSearchList in performance_labels


def test_volcano_milvus_only_diskann_supported():
    """Non-DISKANN index types should not silently fall back to upstream Milvus."""
    assert DB.VolcanoMilvus.case_config_cls(index_type=IndexType.DISKANN) is not None
    assert DB.VolcanoMilvus.case_config_cls(index_type=IndexType.HNSW) is None
    assert DB.VolcanoMilvus.case_config_cls(index_type=IndexType.IVFFlat) is None
