"""Tests for the single db_case_config finalization choke point."""

from types import SimpleNamespace

from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig, ElasticCloudIndexConfig
from vectordb_bench.backend.clients.milvus.config import MilvusIndexConfig
from vectordb_bench.backend.db_case_config import finalize_db_case_config


def _dataset(metric_type: MetricType = MetricType.L2) -> SimpleNamespace:
    return SimpleNamespace(metric_type=metric_type)


def test_finalize_does_not_mutate_the_input_config():
    base = MilvusIndexConfig(index=IndexType.HNSW)

    final = finalize_db_case_config(DB.Milvus, "Performance1536D50K", base, dataset=_dataset())

    assert final is not base
    assert base.metric_type is None
    assert final.metric_type == MetricType.L2


def test_finalize_applies_dataset_metric_on_cli_and_ui_paths_identically():
    base = MilvusIndexConfig(index=IndexType.HNSW)

    cli_result = finalize_db_case_config(DB.Milvus, "Performance1536D50K", base, parameters={}, dataset=_dataset())
    ui_result = finalize_db_case_config(DB.Milvus, "Performance1536D50K", base, dataset=_dataset())

    assert cli_result.metric_type == MetricType.L2
    assert ui_result.metric_type == MetricType.L2


def test_finalize_skips_dataset_metric_for_fts_case_type():
    base = MilvusIndexConfig(index=IndexType.HNSW)

    final = finalize_db_case_config(DB.Milvus, "FTSBm25Performance", base, dataset=_dataset())

    assert final is base
    assert final.metric_type is None


def test_finalize_skips_dataset_metric_for_empty_config():
    base = EmptyDBCaseConfig()

    final = finalize_db_case_config(DB.Milvus, "Performance1536D50K", base, dataset=_dataset())

    assert final is base


def test_finalize_routes_vector_config_to_fts_config_class():
    base = MilvusIndexConfig(index=IndexType.HNSW)

    final = finalize_db_case_config(DB.Milvus, "FTSBm25Performance", base, parameters={})

    assert isinstance(final, DB.Milvus.case_config_cls(IndexType.FTS))
    assert final.bm25_k1 is None


def test_finalize_routes_elastic_cloud_vector_config_to_fts_with_bm25_overrides():
    base = ElasticCloudIndexConfig(index=IndexType.ES_HNSW, number_of_shards=3)

    final = finalize_db_case_config(
        DB.ElasticCloud,
        "FTSBm25Performance",
        base,
        parameters={"bm25_k1": 1.7, "bm25_b": 0.2},
    )

    assert isinstance(final, ElasticCloudFtsConfig)
    assert final.number_of_shards == 3
    assert final.bm25_k1 == 1.7
    assert final.bm25_b == 0.2


def test_finalize_leaves_non_fts_case_configs_unchanged_without_dataset():
    base = MilvusIndexConfig(index=IndexType.HNSW)

    final = finalize_db_case_config(DB.Milvus, "Performance1536D50K", base)

    assert final is base
