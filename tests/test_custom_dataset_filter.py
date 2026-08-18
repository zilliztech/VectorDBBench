import pytest

from vectordb_bench.backend.cases import CaseType, PerformanceCustomDataset
from vectordb_bench.backend.filter import LabelFilter
from vectordb_bench.models import CaseConfig


def _custom_case_kwargs(**overrides):
    kwargs = {
        "name": "custom-ds",
        "description": "",
        "load_timeout": 1.0,
        "optimize_timeout": 1.0,
        "dataset_config": {
            "name": "custom-ds",
            "dir": "/tmp/custom-ds",
            "size": 1000,
            "dim": 128,
            "metric_type": "L2",
            "file_count": 1,
        },
    }
    kwargs.update(overrides)
    return kwargs


def test_custom_dataset_filter_sets_filter_rate_from_label_percentage():
    case = PerformanceCustomDataset(**_custom_case_kwargs(use_filter=True, label_percentage=0.01))
    assert case.filter_rate == pytest.approx(0.99)
    assert isinstance(case.filters, LabelFilter)
    assert case.filters.filter_rate == pytest.approx(0.99)


def test_custom_dataset_filter_matches_label_filter_formula():
    case = PerformanceCustomDataset(**_custom_case_kwargs(use_filter=True, label_percentage=0.2))
    assert case.filter_rate == pytest.approx(1.0 - 0.2)


def test_custom_dataset_without_filter_leaves_filter_rate_unset():
    case = PerformanceCustomDataset(**_custom_case_kwargs(use_filter=False))
    assert case.filter_rate is None


def test_case_config_builds_custom_dataset_with_filter_rate():
    case = CaseConfig(
        case_id=CaseType.PerformanceCustomDataset,
        custom_case=_custom_case_kwargs(use_filter=True, label_percentage=0.5),
    ).case
    assert case.filter_rate == pytest.approx(0.5)
