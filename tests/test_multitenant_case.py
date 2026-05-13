from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vectordb_bench.backend.cases import CaseType, CloudMultiTenantSearchCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig, VectorDB
from vectordb_bench.backend.clients.zilliz_cloud.config import AutoIndexConfig, ZillizCloudConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import DatasetManager, DatasetWithSizeType
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.models import CaseConfig, TaskConfig, TaskStage


def test_multitenant_case_defaults_to_cohere_large_1000_tenants():
    case = CloudMultiTenantSearchCase()

    assert case.case_id == CaseType.CloudMultiTenantSearchCase
    assert case.dataset_with_size_type == DatasetWithSizeType.CohereLarge
    assert case.dataset.data.size == 10_000_000
    assert case.tenant_count == 1000
    assert case.tenant_prefix == "tenant_"
    assert case.tenant_id_width == 4
    assert case.measure_recall is False
    assert case.is_multitenant is True
    assert case.filters.type == FilterOp.NonFilter


def test_multitenant_case_accepts_dataset_and_tenant_count():
    case = CloudMultiTenantSearchCase(
        dataset_with_size_type=DatasetWithSizeType.CohereSmall.value,
        tenant_count=7,
        tenant_prefix="acct_",
        tenant_id_width=2,
        payload_profile=PayloadProfile.VECTOR.value,
        filter_rate=0.01,
    )

    assert case.dataset_with_size_type == DatasetWithSizeType.CohereSmall
    assert case.dataset.data.size == 100_000
    assert case.payload_profile == PayloadProfile.VECTOR
    assert case.estimated_payload_bytes_per_query(k=10) == PayloadProfile.VECTOR.estimated_bytes_per_query(
        k=10,
        dim=case.dataset.data.dim,
    )
    assert case.filters.type == FilterOp.NumGE
    assert case.tenant_count == 7
    assert case.tenant_for_id(0) == "acct_00"
    assert case.tenant_for_id(8) == "acct_01"
    assert case.tenant_labels_for_ids([0, 1, 8, 13]) == ["acct_00", "acct_01", "acct_01", "acct_06"]


def test_case_config_constructs_multitenant_case():
    case = CaseType.CloudMultiTenantSearchCase.case_cls(
        {
            "dataset_with_size_type": DatasetWithSizeType.CohereSmall.value,
            "tenant_count": 5,
            "payload_profile": PayloadProfile.SCALAR_LABEL.value,
            "label_percentage": 0.01,
        }
    )

    assert isinstance(case, CloudMultiTenantSearchCase)
    assert case.payload_profile == PayloadProfile.SCALAR_LABEL
    assert case.filters.type == FilterOp.StrEqual
    assert case.tenant_labels() == ["tenant_0000", "tenant_0001", "tenant_0002", "tenant_0003", "tenant_0004"]


class TenantApiProbeDB(VectorDB):
    name = "TenantApiProbeDB"

    def __init__(self, dim=2, db_config=None, db_case_config=None, collection_name="test", drop_old=False, **kwargs):
        self.insert_calls = []
        self.search_calls = []

    @contextmanager
    def init(self):
        yield

    def insert_embeddings(self, embeddings, metadata, labels_data=None, tenant_labels_data=None, **kwargs):
        self.insert_calls.append((embeddings, metadata, labels_data, tenant_labels_data))
        return len(embeddings), None

    def search_embedding(self, query, k=100, payload_profile=None, tenant=None):
        self.search_calls.append((query, k, payload_profile, tenant))
        return []

    def optimize(self, data_size=None):
        return None


def test_vector_db_accepts_optional_tenant_context():
    db = TenantApiProbeDB(db_case_config=EmptyDBCaseConfig())

    count, err = db.insert_embeddings([[0.1, 0.2]], [42], tenant_labels_data=["tenant_0002"])
    result = db.search_embedding([0.1, 0.2], tenant="tenant_0002")

    assert count == 1
    assert err is None
    assert result == []
    assert db.insert_calls[0][3] == ["tenant_0002"]
    assert db.search_calls[0][3] == "tenant_0002"


def test_search_only_zilliz_multitenant_validates_existing_partition_key_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CloudMultiTenantSearchCase()
    task = TaskConfig(
        db=DB.ZillizCloud,
        db_config=ZillizCloudConfig(uri="http://example.invalid", collection_name="existing"),
        db_case_config=AutoIndexConfig(use_partition_key=False),
        case_config=CaseConfig(case_id=CaseType.CloudMultiTenantSearchCase),
        stages=[TaskStage.SEARCH_CONCURRENT],
    )
    runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    calls: list[tuple[str, object]] = []

    class ExistingCollectionDB:
        def supports_multitenant(self) -> bool:
            return True

        def set_multitenant_context(self, tenant_labels: list[str]) -> None:
            calls.append(("set_context", tenant_labels))

        def validate_multitenant_schema(self) -> None:
            calls.append(("validate_schema", None))

    def fake_init_db(self: CaseRunner, _drop_old: bool = True) -> None:
        self.db = ExistingCollectionDB()

    def fake_prepare(*_args: object, **_kwargs: object) -> None:
        calls.append(("prepare", None))

    monkeypatch.setattr(CaseRunner, "init_db", fake_init_db)
    monkeypatch.setattr(DatasetManager, "prepare", fake_prepare)

    runner._pre_run(drop_old=False)

    assert ("validate_schema", None) in calls
    assert calls[0][0] == "set_context"


def test_zilliz_multitenant_create_still_requires_partition_key() -> None:
    case = CloudMultiTenantSearchCase()
    task = TaskConfig(
        db=DB.ZillizCloud,
        db_config=ZillizCloudConfig(uri="http://example.invalid", collection_name="new_collection"),
        db_case_config=AutoIndexConfig(use_partition_key=False),
        case_config=CaseConfig(case_id=CaseType.CloudMultiTenantSearchCase),
        stages=[TaskStage.DROP_OLD, TaskStage.LOAD],
    )
    runner = CaseRunner(
        run_id="run-id",
        config=task,
        ca=case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )

    with pytest.raises(ValueError, match="requires use_partition_key=True"):
        runner._pre_run(drop_old=True)


class FakeTurboNamespace:
    def __init__(self):
        self.write_calls = []
        self.query_calls = []

    def write(self, **kwargs):
        self.write_calls.append(kwargs)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(rows=[SimpleNamespace(id="10")])

    def metadata(self):
        return {"index": {"unindexed_bytes": 0}}


class FakeTurboClient:
    def __init__(self):
        self.namespaces = {}

    def namespace(self, name):
        self.namespaces.setdefault(name, FakeTurboNamespace())
        return self.namespaces[name]


def test_turbopuffer_groups_multitenant_insert_and_search(monkeypatch):
    from vectordb_bench.backend.clients.api import MetricType
    from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
    from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer

    fake_client = FakeTurboClient()
    monkeypatch.setattr(TurboPuffer, "_create_client", lambda self: fake_client)

    db = TurboPuffer(
        dim=2,
        db_config={
            "api_key": "k",
            "region": "r",
            "api_base_url": None,
            "namespace": "single",
            "multitenant_namespace_prefix": "mt_",
        },
        db_case_config=TurboPufferIndexConfig(metric_type=MetricType.COSINE),
        drop_old=False,
    )
    db.set_multitenant_context(["tenant_0000", "tenant_0001"])

    with db.init():
        count, err = db.insert_embeddings(
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            metadata=[0, 1, 2],
            tenant_labels_data=["tenant_0000", "tenant_0001", "tenant_0000"],
        )
        result = db.search_embedding([1.0, 0.0], k=1, tenant="tenant_0001")

    assert count == 3
    assert err is None
    assert result == [10]
    assert fake_client.namespaces["mt_tenant_0000"].write_calls[0]["upsert_columns"]["id"] == [0, 2]
    assert fake_client.namespaces["mt_tenant_0001"].write_calls[0]["upsert_columns"]["id"] == [1]
    assert fake_client.namespaces["mt_tenant_0001"].query_calls
