import pytest

from vectordb_bench.backend.cases import FTS_FILTER_ID_FIELD, FTS_FILTER_RATES, FTSBm25Performance
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.frontend.config.dbCaseConfigs import generate_fts_case, get_fts_case_items


def test_fts_filter_case_uses_filter_id_for_large_dataset():
    case = FTSBm25Performance(
        dataset_with_size_type=FtsDatasetWithSizeType.MSMarcoLarge,
        filter_rate=0.95,
    )

    filters = case.filters

    assert filters.type == FilterOp.NumGE
    assert filters.int_field == FTS_FILTER_ID_FIELD
    assert filters.int_value == int(8_841_823 * 0.95)
    assert "Filter 95%" in case.name


def test_fts_filter_case_rejects_small_and_medium_datasets():
    with pytest.raises(ValueError, match="only supported"):
        FTSBm25Performance(
            dataset_with_size_type=FtsDatasetWithSizeType.MSMarcoSmall,
            filter_rate=0.5,
        )

    with pytest.raises(ValueError, match="only supported"):
        FTSBm25Performance(
            dataset_with_size_type=FtsDatasetWithSizeType.HotpotQAMedium,
            filter_rate=0.5,
        )


def test_fts_filter_case_rejects_unsupported_filter_rate():
    with pytest.raises(ValueError, match="must be one of"):
        FTSBm25Performance(
            dataset_with_size_type=FtsDatasetWithSizeType.HotpotQALarge,
            filter_rate=0.8,
        )


def test_frontend_generates_fts_filter_cases_only_for_large_datasets():
    items = get_fts_case_items()
    filtered_cases = [
        case
        for item in items
        for case in item.cases
        if case.custom_case and case.custom_case.get("filter_rate") is not None
    ]

    assert len(filtered_cases) == 2 * len(FTS_FILTER_RATES)
    assert {case.custom_case["filter_rate"] for case in filtered_cases} == set(FTS_FILTER_RATES)
    assert {
        case.custom_case["dataset_with_size_type"]
        for case in filtered_cases
    } == {
        FtsDatasetWithSizeType.MSMarcoLarge.value,
        FtsDatasetWithSizeType.HotpotQALarge.value,
    }


def test_frontend_generate_fts_case_adds_filter_rate_when_requested():
    case_config = generate_fts_case(FtsDatasetWithSizeType.HotpotQALarge, filter_rate=0.99)

    assert case_config.custom_case == {
        "dataset_with_size_type": FtsDatasetWithSizeType.HotpotQALarge.value,
        "filter_rate": 0.99,
    }
