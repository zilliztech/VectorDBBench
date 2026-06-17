import pytest
from click.testing import CliRunner

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.aliyun_milvus import cli as aliyun_milvus_cli
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.milvus.config import MilvusConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.frontend.config.dbCaseConfigs import (
    AliyunMilvusLoadConfig,
    AliyunMilvusPerformanceConfig,
)
from vectordb_bench.models import CaseConfigParamType


def _diskann_config_cls():
    return DB.AliyunMilvus.case_config_cls(index_type=IndexType.DISKANN)


def test_aliyun_milvus_reuses_upstream_milvus_client_and_config():
    """AliyunMilvus is minimal: it reuses the upstream Milvus client and config."""
    assert DB.AliyunMilvus.init_cls is Milvus
    assert DB.AliyunMilvus.config_cls is MilvusConfig


def test_aliyun_milvus_search_params_opt_in_by_default():
    """With none of the three knobs set, only search_list is sent (same as Milvus DISKANN)."""
    case_config = _diskann_config_cls()(search_list=200)

    assert case_config.search_param()["params"] == {"search_list": 200}


def test_aliyun_milvus_search_params_injected_when_set():
    case_config = _diskann_config_cls()(
        search_list=200,
        rerank_topk_multiplier=0,
        early_termination_threshold=0,
        cross_segment_rerank=True,
    )

    assert case_config.search_param()["params"] == {
        "search_list": 200,
        "rerank_topk_multiplier": 0,
        "early_termination_threshold": 0,
        "cross_segment_rerank": True,
    }


def test_aliyun_milvus_zero_is_a_meaningful_value_not_unset():
    """0 is a real value (e.g. disables rerank reads) and must be sent."""
    case_config = _diskann_config_cls()(search_list=200, rerank_topk_multiplier=0)

    assert case_config.search_param()["params"]["rerank_topk_multiplier"] == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("anything", False),
        (1, True),
        (0, False),
    ],
)
def test_aliyun_milvus_cross_segment_rerank_normalizes_non_bool_inputs(raw, expected):
    """CLI/UI may pass strings or ints for the bool knob; coerce them consistently."""
    case_config = _diskann_config_cls()(search_list=200, cross_segment_rerank=raw)

    assert case_config.cross_segment_rerank is expected
    assert case_config.search_param()["params"]["cross_segment_rerank"] is expected


def test_aliyun_milvus_ui_sentinels_normalize_to_unset():
    """UI 'unset' sentinels (-1 for numbers, 'DEFAULT' for the bool) -> None -> omitted."""
    case_config = _diskann_config_cls()(
        search_list=200,
        rerank_topk_multiplier=-1,
        early_termination_threshold=-1,
        cross_segment_rerank="DEFAULT",
    )

    assert case_config.search_param()["params"] == {"search_list": 200}


def test_aliyun_milvus_frontend_exposes_search_params_in_performance_only():
    """The three knobs are search-time only: shown in Performance, not in Load."""
    load_labels = [config.label for config in AliyunMilvusLoadConfig]
    performance_labels = [config.label for config in AliyunMilvusPerformanceConfig]

    for label in (
        CaseConfigParamType.rerank_topk_multiplier,
        CaseConfigParamType.early_termination_threshold,
        CaseConfigParamType.cross_segment_rerank,
    ):
        assert label in performance_labels
        assert label not in load_labels


def test_aliyun_milvus_load_config_has_no_search_list():
    """search_list is a query-time param, must not appear in load config."""
    load_labels = [config.label for config in AliyunMilvusLoadConfig]
    assert CaseConfigParamType.SearchList not in load_labels


def test_aliyun_milvus_only_diskann_supported():
    """Non-DISKANN index types should not silently fall back to upstream Milvus."""
    assert DB.AliyunMilvus.case_config_cls(index_type=IndexType.DISKANN) is not None
    assert DB.AliyunMilvus.case_config_cls(index_type=IndexType.HNSW) is None
    assert DB.AliyunMilvus.case_config_cls(index_type=IndexType.IVFFlat) is None


def test_aliyun_milvus_cli_omitting_knobs_sends_only_search_list(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    monkeypatch.setattr(aliyun_milvus_cli, "run", lambda **kwargs: captured.update(kwargs))

    runner = CliRunner()
    result = runner.invoke(
        aliyun_milvus_cli.AliyunMilvusDISKANN,
        ["--uri", "http://localhost:19530", "--search-list", "200"],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["db_case_config"]
    assert case_config.rerank_topk_multiplier is None
    assert case_config.early_termination_threshold is None
    assert case_config.cross_segment_rerank is None
    assert case_config.search_param()["params"] == {"search_list": 200}


def test_aliyun_milvus_cli_passes_knobs_when_specified(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    monkeypatch.setattr(aliyun_milvus_cli, "run", lambda **kwargs: captured.update(kwargs))

    runner = CliRunner()
    result = runner.invoke(
        aliyun_milvus_cli.AliyunMilvusDISKANN,
        [
            "--uri", "http://localhost:19530",
            "--search-list", "200",
            "--rerank-topk-multiplier", "0",
            "--early-termination-threshold", "0",
            "--cross-segment-rerank",
        ],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["db_case_config"]
    assert case_config.rerank_topk_multiplier == 0
    assert case_config.early_termination_threshold == 0
    assert case_config.cross_segment_rerank is True


def test_aliyun_milvus_cli_no_cross_segment_rerank_sends_false(monkeypatch: pytest.MonkeyPatch):
    """--no-cross-segment-rerank explicitly disables it: send False, not omit."""
    captured = {}

    monkeypatch.setattr(aliyun_milvus_cli, "run", lambda **kwargs: captured.update(kwargs))

    runner = CliRunner()
    result = runner.invoke(
        aliyun_milvus_cli.AliyunMilvusDISKANN,
        [
            "--uri", "http://localhost:19530",
            "--search-list", "200",
            "--no-cross-segment-rerank",
        ],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["db_case_config"]
    assert case_config.cross_segment_rerank is False
    assert case_config.search_param()["params"] == {
        "search_list": 200,
        "cross_segment_rerank": False,
    }
