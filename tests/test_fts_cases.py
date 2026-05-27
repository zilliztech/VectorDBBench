from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType
from vectordb_bench.cli.cli import CommonTypedDict, get_custom_case_config
from vectordb_bench.frontend.config.dbCaseConfigs import (
    CASE_CONFIG_MAP,
    UI_CASE_CLUSTERS,
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
        }
    )

    assert custom_case_config == {
        "dataset_with_size_type": FtsDatasetWithSizeType.HotpotQAMedium.value,
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
    }


def test_cli_dataset_option_help_includes_default_fts_datasets():
    dataset_option = CommonTypedDict.__annotations__["dataset_with_size_type"].__metadata__[0]
    option_kwargs = dataset_option.__closure__[0].cell_contents

    assert FtsDatasetWithSizeType.MSMarcoSmall.value in option_kwargs["help"]
    assert FtsDatasetWithSizeType.MSMarcoMedium.value in option_kwargs["help"]
    assert FtsDatasetWithSizeType.HotpotQASmall.value in option_kwargs["help"]
    assert FtsDatasetWithSizeType.HotpotQAMedium.value in option_kwargs["help"]
