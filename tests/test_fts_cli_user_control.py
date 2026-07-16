from typing import get_type_hints

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig, ElasticCloudIndexConfig
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig, VespaHNSWConfig
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType, FtsFilterIdDistribution
from vectordb_bench.cli.cli import CommonTypedDict, get_custom_case_config, select_cli_db_case_config


def test_common_cli_exposes_optional_fts_bm25_overrides():
    hints = get_type_hints(CommonTypedDict, include_extras=True)

    assert "bm25_k1" in hints
    assert "bm25_b" in hints
    assert "fts_filter_rate" in hints
    assert "fts_filter_id_distribution" in hints


def test_cli_builds_fts_filter_case_config():
    custom_case = get_custom_case_config(
        {
            "case_type": "FTSBm25Performance",
            "dataset_with_size_type": FtsDatasetWithSizeType.MSMarcoLarge.value,
            "payload_profile": "ids_only",
            "fts_filter_rate": 0.95,
        }
    )

    assert custom_case == {
        "dataset_with_size_type": FtsDatasetWithSizeType.MSMarcoLarge.value,
        "payload_profile": "ids_only",
        "filter_rate": 0.95,
        "filter_id_distribution": FtsFilterIdDistribution.Permuted.value,
    }


def test_cli_builds_sequential_fts_filter_case_config():
    custom_case = get_custom_case_config(
        {
            "case_type": "FTSBm25Performance",
            "dataset_with_size_type": FtsDatasetWithSizeType.HotpotQALarge.value,
            "fts_filter_rate": 0.99,
            "fts_filter_id_distribution": FtsFilterIdDistribution.Sequential.value,
        }
    )

    assert custom_case["filter_id_distribution"] == FtsFilterIdDistribution.Sequential.value


def test_cli_applies_bm25_overrides_to_existing_fts_config():
    config = MilvusFtsConfig(drop_ratio_search=0.1)

    selected = select_cli_db_case_config(
        DB.Milvus,
        config,
        "FTSBm25Performance",
        {"bm25_k1": 1.4, "bm25_b": 0.6},
    )

    assert isinstance(selected, MilvusFtsConfig)
    assert selected.bm25_k1 == 1.4
    assert selected.bm25_b == 0.6
    assert selected.drop_ratio_search == 0.1
    assert selected.sparse_index_param()["params"] == {
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.4,
        "bm25_b": 0.6,
    }


def test_cli_applies_bm25_overrides_after_routing_vector_config_to_fts():
    selected = select_cli_db_case_config(
        DB.ElasticCloud,
        ElasticCloudIndexConfig(index=IndexType.ES_HNSW, number_of_shards=3),
        "FTSBm25Performance",
        {"bm25_k1": 1.7, "bm25_b": 0.2},
    )

    assert isinstance(selected, ElasticCloudFtsConfig)
    assert selected.number_of_shards == 3
    assert selected.bm25_k1 == 1.7
    assert selected.bm25_b == 0.2
    assert selected.index_param()["properties"]["text"]["similarity"] == "vdbbench_bm25"
    assert selected.similarity_settings() == {
        "similarity": {
            "vdbbench_bm25": {
                "type": "BM25",
                "k1": 1.7,
                "b": 0.2,
            }
        }
    }


def test_cli_leaves_fts_bm25_defaults_when_options_are_omitted():
    selected = select_cli_db_case_config(
        DB.Vespa,
        VespaHNSWConfig(),
        "FTSBm25Performance",
        {"bm25_k1": None, "bm25_b": None},
    )

    assert isinstance(selected, VespaFtsConfig)
    assert selected.bm25_k1 is None
    assert selected.bm25_b is None
    assert selected.rank_properties() == []
