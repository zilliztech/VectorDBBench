import pytest

from vectordb_bench import config
from vectordb_bench.backend.cases import CaseType, CloudPayloadSearchCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.models import CaseConfig, TaskConfig


class FakeDB:
    name = "FakeDB"

    def __init__(self, supported_payload_profiles=None):
        self.supported_payload_profiles = supported_payload_profiles or {PayloadProfile.IDS_ONLY}
        self.calls = []

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile in self.supported_payload_profiles

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        self.calls.append((query, k, kwargs))
        return list(range(k))


def test_payload_profile_estimates_response_bytes():
    assert PayloadProfile.IDS_ONLY.estimated_bytes_per_query(k=10, dim=768) == 200
    assert PayloadProfile.VECTOR.estimated_bytes_per_query(k=10, dim=768) == 30_920


def test_cloud_payload_case_defaults_to_laion_100m():
    case = CloudPayloadSearchCase(payload_profile="vector")

    assert case.case_id == CaseType.CloudPayloadSearchCase
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000
    assert case.dataset.data.dim == 768
    assert case.payload_profile == PayloadProfile.VECTOR
    assert case.estimated_payload_bytes_per_query(config.K_DEFAULT) == 309_200


def test_cloud_payload_case_accepts_sized_dataset():
    case = CloudPayloadSearchCase(dataset_with_size_type=DatasetWithSizeType.CohereSmall.value)

    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.name == "Cohere"
    assert case.dataset.data.size == 100_000


def test_case_config_builds_cloud_payload_case_from_custom_case():
    case = CaseConfig(
        case_id=CaseType.CloudPayloadSearchCase,
        custom_case={"payload_profile": "vector"},
    ).case

    assert isinstance(case, CloudPayloadSearchCase)
    assert case.payload_profile == PayloadProfile.VECTOR


def test_case_runner_reuse_key_distinguishes_scalar_label_schema_requirement():
    ids_only_case = CloudPayloadSearchCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        payload_profile=PayloadProfile.IDS_ONLY,
    )
    scalar_label_case = CloudPayloadSearchCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        payload_profile=PayloadProfile.SCALAR_LABEL,
    )
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(case_id=CaseType.CloudPayloadSearchCase),
    )

    ids_only_runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=ids_only_case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    scalar_label_runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=scalar_label_case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )

    assert ids_only_case.with_scalar_labels is False
    assert scalar_label_case.with_scalar_labels is True
    assert ids_only_runner != scalar_label_runner
    assert hash(ids_only_runner) != hash(scalar_label_runner)


def test_cli_propagates_cloud_payload_dataset_selection():
    custom_case = get_custom_case_config(
        {
            "case_type": "CloudPayloadSearchCase",
            "dataset_with_size_type": DatasetWithSizeType.CohereSmall.value,
            "payload_profile": "vector",
            "cloud_filter_rate": None,
            "cloud_label_percentage": None,
        }
    )

    assert custom_case["dataset_with_size_type"] == DatasetWithSizeType.CohereSmall.value

    case = CaseConfig(case_id=CaseType.CloudPayloadSearchCase, custom_case=custom_case).case
    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.size == 100_000


def test_cli_omits_cloud_payload_dataset_when_not_selected():
    custom_case = get_custom_case_config(
        {
            "case_type": "CloudPayloadSearchCase",
            "dataset_with_size_type": None,
            "payload_profile": "ids_only",
            "cloud_filter_rate": None,
            "cloud_label_percentage": None,
        }
    )

    assert "dataset_with_size_type" not in custom_case

    case = CaseConfig(case_id=CaseType.CloudPayloadSearchCase, custom_case=custom_case).case
    assert case.dataset_with_size_type is None
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000


def test_serial_runner_omits_payload_argument_for_ids_only():
    db = FakeDB()
    runner = SerialSearchRunner(db=db, test_data=[[0.1]], ground_truth=[[0]], k=3)

    assert runner._get_db_search_res([0.1]) == [0, 1, 2]
    assert db.calls == [([0.1], 3, {})]


def test_serial_runner_passes_payload_argument_for_vector_profile():
    db = FakeDB(supported_payload_profiles={PayloadProfile.IDS_ONLY, PayloadProfile.VECTOR})
    runner = SerialSearchRunner(
        db=db,
        test_data=[[0.1]],
        ground_truth=[[0]],
        k=3,
        payload_profile=PayloadProfile.VECTOR,
    )

    assert runner._get_db_search_res([0.1]) == [0, 1, 2]
    assert db.calls == [([0.1], 3, {"payload_profile": PayloadProfile.VECTOR})]


def test_search_runners_fail_fast_for_unsupported_payload_profile():
    db = FakeDB()

    with pytest.raises(NotImplementedError, match="payload_profile=vector"):
        SerialSearchRunner(
            db=db,
            test_data=[[0.1]],
            ground_truth=[[0]],
            k=3,
            payload_profile=PayloadProfile.VECTOR,
        )

    with pytest.raises(NotImplementedError, match="payload_profile=vector"):
        MultiProcessingSearchRunner(
            db=db,
            test_data=[[0.1]],
            k=3,
            payload_profile=PayloadProfile.VECTOR,
        )
