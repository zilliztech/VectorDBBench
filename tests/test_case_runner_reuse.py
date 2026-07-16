from pydantic import SecretStr

from vectordb_bench import config
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig, MetricType
from vectordb_bench.backend.clients.doris.config import DorisCaseConfig, DorisConfig
from vectordb_bench.backend.clients.pinecone.config import PineconeConfig
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferConfig, TurboPufferIndexConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus, TaskRunner
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseType, TaskConfig, TaskStage, TestResult

DEFAULT_INSERT_BATCH_SIZE = config.DEFAULT_INSERT_BATCH_SIZE


def make_runner(
    *,
    case_id: CaseType = CaseType.Performance1536D50K,
    custom_case: dict | None = None,
    db: DB = DB.TurboPuffer,
    db_config=None,
    db_case_config=None,
    stages: list[TaskStage] | None = None,
    insert_batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
) -> CaseRunner:
    if db_config is None:
        if db == DB.TurboPuffer:
            db_config = TurboPufferConfig(api_key="key", region="aws-us-east-1")
        elif db == DB.Pinecone:
            db_config = PineconeConfig(api_key="key", index_name="idx")
        elif db == DB.Doris:
            db_config = DorisConfig(password=SecretStr(""))
        else:
            db_config = DB.Test.config_cls()
    if db_case_config is None:
        if db == DB.TurboPuffer:
            db_case_config = TurboPufferIndexConfig(metric_type=MetricType.COSINE)
        elif db == DB.Doris:
            db_case_config = DorisCaseConfig(metric_type=MetricType.COSINE)
        else:
            db_case_config = EmptyDBCaseConfig()

    task = TaskConfig(
        db=db,
        db_config=db_config,
        db_case_config=db_case_config,
        case_config=CaseConfig(case_id=case_id, custom_case=custom_case or {}),
        stages=stages or [TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_SERIAL],
        insert_batch_size=insert_batch_size,
    )
    return CaseRunner(
        run_id="run-id",
        config=task,
        ca=task.case_config.case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )


def assert_not_reusable(left: CaseRunner, right: CaseRunner) -> None:
    assert left != right
    assert right != left
    assert hash(left) != hash(right)


def test_reuse_key_distinguishes_multitenant_routing_parameters():
    base_case = {
        "dataset_with_size_type": DatasetWithSizeType.CohereSmall.value,
        "tenant_count": 2,
        "tenant_prefix": "tenant_",
        "tenant_id_width": 4,
    }

    assert_not_reusable(
        make_runner(case_id=CaseType.CloudMultiTenantSearchCase, custom_case=base_case),
        make_runner(case_id=CaseType.CloudMultiTenantSearchCase, custom_case={**base_case, "tenant_count": 3}),
    )
    assert_not_reusable(
        make_runner(case_id=CaseType.CloudMultiTenantSearchCase, custom_case=base_case),
        make_runner(case_id=CaseType.CloudMultiTenantSearchCase, custom_case={**base_case, "tenant_prefix": "org_"}),
    )
    assert_not_reusable(
        make_runner(case_id=CaseType.CloudMultiTenantSearchCase, custom_case=base_case),
        make_runner(case_id=CaseType.CloudMultiTenantSearchCase, custom_case={**base_case, "tenant_id_width": 2}),
    )


def test_reuse_key_distinguishes_single_layout_from_multitenant_layout():
    dataset_case = {"dataset_with_size_type": DatasetWithSizeType.CohereSmall.value}

    assert_not_reusable(
        make_runner(case_id=CaseType.CloudPayloadSearchCase, custom_case=dataset_case),
        make_runner(
            case_id=CaseType.CloudMultiTenantSearchCase,
            custom_case={**dataset_case, "tenant_count": 2},
        ),
    )


def test_reuse_key_preserves_safe_payload_reuse():
    dataset_case = {"dataset_with_size_type": DatasetWithSizeType.CohereSmall.value}

    ids_only = make_runner(
        case_id=CaseType.CloudPayloadSearchCase,
        custom_case={**dataset_case, "payload_profile": "ids_only"},
    )
    vector = make_runner(
        case_id=CaseType.CloudPayloadSearchCase,
        custom_case={**dataset_case, "payload_profile": "vector"},
    )

    assert ids_only == vector
    assert hash(ids_only) == hash(vector)


def test_reuse_key_distinguishes_insert_batch_size():
    assert_not_reusable(
        make_runner(insert_batch_size=100),
        make_runner(insert_batch_size=200),
    )


def test_reuse_key_distinguishes_physical_db_targets():
    assert_not_reusable(
        make_runner(db_config=TurboPufferConfig(api_key="key", region="aws-us-east-1", namespace="namespace_a")),
        make_runner(db_config=TurboPufferConfig(api_key="key", region="aws-us-east-1", namespace="namespace_b")),
    )
    assert_not_reusable(
        make_runner(db=DB.Pinecone, db_config=PineconeConfig(api_key="key", index_name="index_a")),
        make_runner(db=DB.Pinecone, db_config=PineconeConfig(api_key="key", index_name="index_b")),
    )


def test_reuse_key_distinguishes_doris_case_derived_table_names():
    assert_not_reusable(
        make_runner(db=DB.Doris, case_id=CaseType.Performance768D1M),
        make_runner(
            db=DB.Doris,
            case_id=CaseType.NewIntFilterPerformanceCase,
            custom_case={
                "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
                "filter_rate": 0.01,
            },
        ),
    )


def test_search_only_runner_does_not_suppress_later_full_load(monkeypatch):
    calls: list[bool] = []
    search_only = make_runner(
        stages=[TaskStage.SEARCH_SERIAL],
    )
    full_load = make_runner(
        stages=[TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_SERIAL],
    )

    def fake_run(self: CaseRunner, drop_old: bool = True) -> Metric:
        calls.append(drop_old)
        return Metric()

    class SendConn:
        def __init__(self):
            self.sent = []

        def send(self, value):
            self.sent.append(value)

        def close(self):
            return None

    monkeypatch.setattr(CaseRunner, "run", fake_run)
    monkeypatch.setattr(TestResult, "display", lambda self: None)
    monkeypatch.setattr(TestResult, "flush", lambda self: None)

    BenchMarkRunner()._async_task_v2(
        TaskRunner(run_id="run-id", task_label="task", case_runners=[search_only, full_load]),
        SendConn(),
    )

    assert calls == [False, True]
