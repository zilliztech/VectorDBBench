from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudFtsConfig, ElasticCloudIndexConfig
from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferFtsConfig, TurboPufferIndexConfig
from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig, VespaHNSWConfig
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.cli.cli import CommonTypedDict, get_custom_case_config, select_cli_db_case_config
from vectordb_bench.frontend.config.dbCaseConfigs import (
    CASE_CONFIG_MAP,
    UI_CASE_CLUSTERS,
    CaseConfigParamInput_FTS_analyzer_max_token_length,
    get_case_config_inputs,
    get_fts_case_items,
    get_selectable_case_items,
)
from vectordb_bench.models import CaseConfig


def test_fts_case_defaults_to_msmarco_small():
    case = CaseConfig(case_id=CaseType.FTSmsmarcoPerformance).case

    assert case.label == CaseLabel.FullTextSearchPerformance
    assert case.dataset_with_size_type == FtsDatasetWithSizeType.MSMarcoSmall
    assert case.dataset.data.size == 100_000
    assert "MS MARCO Small" in case.name


def test_fts_case_accepts_hotpotqa_medium():
    case = CaseConfig(
        case_id=CaseType.FTSmsmarcoPerformance,
        custom_case={"dataset_with_size_type": FtsDatasetWithSizeType.HotpotQAMedium.value},
    ).case

    assert case.dataset_with_size_type == FtsDatasetWithSizeType.HotpotQAMedium
    assert case.dataset.data.name == "HotpotQA"
    assert case.dataset.data.size == 1_000_000
    assert "HotpotQA Medium" in case.name


def test_fts_case_payload_estimate_does_not_require_vector_dim():
    case = CaseConfig(case_id=CaseType.FTSmsmarcoPerformance).case

    assert not hasattr(case.dataset.data, "dim")
    assert case.payload_profile == PayloadProfile.IDS_ONLY
    assert case.estimated_payload_bytes_per_query(k=100) == 2_000


def test_fts_case_accepts_text_payload_profile():
    case = CaseConfig(
        case_id=CaseType.FTSmsmarcoPerformance,
        custom_case={"payload_profile": PayloadProfile.TEXT.value},
    ).case

    assert case.payload_profile == PayloadProfile.TEXT
    assert case.estimated_payload_bytes_per_query(k=10) == 5_320


def test_fts_case_items_expose_small_and_medium_by_default():
    case_items = get_fts_case_items()
    dataset_values = [
        case.custom_case["dataset_with_size_type"]
        for item in case_items
        for case in item.cases
    ]

    assert dataset_values == [
        FtsDatasetWithSizeType.MSMarcoSmall.value,
        FtsDatasetWithSizeType.MSMarcoMedium.value,
        FtsDatasetWithSizeType.HotpotQASmall.value,
        FtsDatasetWithSizeType.HotpotQAMedium.value,
    ]
    assert FtsDatasetWithSizeType.MSMarcoLarge.value not in dataset_values
    assert FtsDatasetWithSizeType.HotpotQALarge.value not in dataset_values
    assert all(item.caseLabel == CaseLabel.FullTextSearchPerformance for item in case_items)


def test_fts_case_items_can_include_advanced_large_cases():
    dataset_values = [
        case.custom_case["dataset_with_size_type"]
        for item in get_fts_case_items(include_advanced=True)
        for case in item.cases
    ]

    assert FtsDatasetWithSizeType.MSMarcoLarge.value in dataset_values
    assert FtsDatasetWithSizeType.HotpotQALarge.value in dataset_values


def test_case_config_map_exposes_fts_only_for_supported_backends():
    supported_backends = {
        DB.Milvus,
        DB.ElasticCloud,
        DB.Vespa,
        DB.TurboPuffer,
    }
    unsupported_backends = set(DB) - supported_backends - {DB.Test}

    for db in supported_backends:
        assert CaseLabel.FullTextSearchPerformance in CASE_CONFIG_MAP[db]

    for db in unsupported_backends:
        assert CaseLabel.FullTextSearchPerformance not in CASE_CONFIG_MAP.get(db, {})


def test_turbopuffer_missing_vector_ui_configs_return_empty_inputs():
    assert get_case_config_inputs(DB.TurboPuffer, CaseLabel.Load) == []
    assert get_case_config_inputs(DB.TurboPuffer, CaseLabel.Performance) == []


def test_milvus_fts_max_token_length_ui_uses_config_field_name():
    assert CaseConfigParamInput_FTS_analyzer_max_token_length.label.value == "analyzer_max_token_length"


def test_fts_ui_cases_are_selectable_only_when_active_backends_support_fts():
    fts_cluster = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Full-Text Search (FTS) Test")

    assert get_selectable_case_items(fts_cluster, [DB.Milvus])
    assert get_selectable_case_items(fts_cluster, [DB.ElasticCloud, DB.Vespa])
    assert get_selectable_case_items(fts_cluster, [DB.Clickhouse]) == []
    assert get_selectable_case_items(fts_cluster, [DB.Milvus, DB.Clickhouse]) == []


def test_cli_custom_case_config_passes_fts_dataset_with_size_type():
    custom_case_config = get_custom_case_config(
        {
            "case_type": "FTSmsmarcoPerformance",
            "dataset_with_size_type": FtsDatasetWithSizeType.HotpotQAMedium.value,
            "payload_profile": "text",
        }
    )

    assert custom_case_config == {
        "dataset_with_size_type": FtsDatasetWithSizeType.HotpotQAMedium.value,
        "payload_profile": "text",
    }


def test_cli_custom_case_config_defaults_to_msmarco_small_for_fts():
    custom_case_config = get_custom_case_config(
        {
            "case_type": "FTSmsmarcoPerformance",
            "dataset_with_size_type": "Medium Cohere (768dim, 1M)",
        }
    )

    assert custom_case_config == {
        "dataset_with_size_type": FtsDatasetWithSizeType.MSMarcoSmall.value,
        "payload_profile": "ids_only",
    }


def test_cli_dataset_option_help_includes_default_fts_datasets():
    dataset_option = CommonTypedDict.__annotations__["dataset_with_size_type"].__metadata__[0]
    option_kwargs = dataset_option.__closure__[0].cell_contents

    assert FtsDatasetWithSizeType.MSMarcoSmall.value in option_kwargs["help"]
    assert FtsDatasetWithSizeType.MSMarcoMedium.value in option_kwargs["help"]
    assert FtsDatasetWithSizeType.HotpotQASmall.value in option_kwargs["help"]
    assert FtsDatasetWithSizeType.HotpotQAMedium.value in option_kwargs["help"]


def test_cli_selects_fts_case_config_for_supported_backends():
    assert isinstance(
        select_cli_db_case_config(
            DB.ElasticCloud,
            ElasticCloudIndexConfig(index=IndexType.ES_HNSW),
            "FTSmsmarcoPerformance",
        ),
        ElasticCloudFtsConfig,
    )
    assert isinstance(select_cli_db_case_config(DB.Vespa, VespaHNSWConfig(), "FTSmsmarcoPerformance"), VespaFtsConfig)
    assert isinstance(
        select_cli_db_case_config(DB.TurboPuffer, TurboPufferIndexConfig(), "FTSmsmarcoPerformance"),
        TurboPufferFtsConfig,
    )


def test_cli_keeps_existing_fts_config_and_vector_config():
    milvus_fts_config = MilvusFtsConfig(drop_ratio_search=0.2)
    assert select_cli_db_case_config(DB.Milvus, milvus_fts_config, "FTSmsmarcoPerformance") is milvus_fts_config

    elastic_vector_config = ElasticCloudIndexConfig(index=IndexType.ES_HNSW)
    assert select_cli_db_case_config(DB.ElasticCloud, elastic_vector_config, "Performance1536D50K") is elastic_vector_config
