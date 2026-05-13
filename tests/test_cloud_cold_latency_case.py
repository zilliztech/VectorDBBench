import pytest

from vectordb_bench.backend.cases import CaseLabel, CaseType, CloudColdLatencyCase
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.models import CaseConfig


def test_cloud_cold_latency_case_defaults_to_laion_100m():
    case = CloudColdLatencyCase()

    assert case.case_id == CaseType.CloudColdLatencyCase
    assert case.label == CaseLabel.CloudColdLatency
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000
    assert case.dataset.data.dim == 768
    assert case.payload_profile == PayloadProfile.IDS_ONLY
    assert case.query_count == 1000
    assert case.filters.type == FilterOp.NonFilter


def test_cloud_cold_latency_case_accepts_payload_dataset_and_int_filter():
    case = CloudColdLatencyCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        payload_profile="vector",
        filter_rate=0.9,
        query_count=10,
    )

    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.name == "Cohere"
    assert case.payload_profile == PayloadProfile.VECTOR
    assert case.filter_rate == 0.9
    assert case.query_count == 10
    assert case.filters.type == FilterOp.NumGE


def test_cloud_cold_latency_case_accepts_label_filter():
    case = CloudColdLatencyCase(label_percentage=0.9)

    assert case.label_percentage == 0.9
    assert case.filters.type == FilterOp.StrEqual


def test_cloud_cold_latency_case_rejects_two_filter_types():
    with pytest.raises(ValueError, match="supports only one filter type"):
        CloudColdLatencyCase(filter_rate=0.9, label_percentage=0.9)


def test_cloud_cold_latency_case_rejects_invalid_query_count():
    with pytest.raises(ValueError, match="query_count must be positive"):
        CloudColdLatencyCase(query_count=0)


def test_case_config_builds_cloud_cold_latency_case_from_custom_case():
    case = CaseConfig(
        case_id=CaseType.CloudColdLatencyCase,
        custom_case={
            "payload_profile": "scalar_label",
            "label_percentage": 0.9,
            "query_count": 12,
        },
    ).case

    assert isinstance(case, CloudColdLatencyCase)
    assert case.payload_profile == PayloadProfile.SCALAR_LABEL
    assert case.label_percentage == 0.9
    assert case.query_count == 12


def test_cli_builds_cloud_cold_latency_custom_case_config():
    params = {
        "case_type": "CloudColdLatencyCase",
        "payload_profile": "vector",
        "cloud_filter_rate": 0.9,
        "cloud_label_percentage": None,
        "cloud_cold_query_count": 1000,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }

    assert get_custom_case_config(params) == {
        "payload_profile": "vector",
        "filter_rate": 0.9,
        "query_count": 1000,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }
