from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.dataset import FtsDatasetWithSizeType
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
