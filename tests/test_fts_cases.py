import pytest

from vectordb_bench.backend.cases import FTS_FILTER_ID_FIELD, FTSBm25Performance
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType
from vectordb_bench.backend.filter import FilterOp


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
