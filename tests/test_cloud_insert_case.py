import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import CaseLabel, CaseType, CloudInsertCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig, VectorDB
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.clients.pinecone.config import PineconeConfig
from vectordb_bench.backend.clients.pinecone.pinecone import Pinecone
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer
from vectordb_bench.backend.clients.zilliz_cloud.config import AutoIndexConfig, ZillizCloudConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig, TaskStage, TestResult


def test_cloud_insert_case_defaults_to_laion_100m():
    case = CloudInsertCase(batch_size=1000)

    assert case.case_id == CaseType.CloudInsertCase
    assert case.label == CaseLabel.CloudInsert
    assert case.dataset.data.name == "LAION"
    assert case.dataset.data.size == 100_000_000
    assert case.batch_size == 1000
    assert case.duration is None


def test_case_config_builds_cloud_insert_case_from_custom_case():
    case = CaseConfig(
        case_id=CaseType.CloudInsertCase,
        custom_case={
            "batch_size": 5000,
            "duration": 1800,
            "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
        },
    ).case

    assert isinstance(case, CloudInsertCase)
    assert case.batch_size == 5000
    assert case.duration == 1800
    assert case.dataset.data.name == "Cohere"
    assert case.dataset.data.size == 1_000_000


def test_cli_builds_cloud_insert_custom_case_config():
    params = {
        "case_type": "CloudInsertCase",
        "cloud_insert_batch_size": 10_000,
        "cloud_insert_duration": 1800,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }

    assert get_custom_case_config(params) == {
        "batch_size": 10_000,
        "duration": 1800,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }


def test_cli_builds_cloud_insert_custom_case_config_with_default_dataset():
    cfg = get_custom_case_config(
        {
            "case_type": "CloudInsertCase",
            "cloud_insert_batch_size": 10_000,
            "cloud_insert_duration": None,
            "dataset_with_size_type": None,
        }
    )

    assert cfg == {
        "batch_size": 10_000,
        "duration": None,
        "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
    }

    case = CaseConfig(case_id=CaseType.CloudInsertCase, custom_case=cfg).case
    assert case.dataset_with_size_type == DatasetWithSizeType.CohereMedium
    assert case.dataset.data.size == 1_000_000


def test_cli_builds_multitenant_custom_case_config():
    cfg = get_custom_case_config(
        {
            "case_type": "CloudMultiTenantSearchCase",
            "dataset_with_size_type": "Small Cohere (768dim, 100K)",
            "tenant_count": 13,
            "tenant_prefix": "acct_",
            "tenant_id_width": 3,
            "payload_profile": "vector",
            "cloud_filter_rate": 0.01,
            "cloud_label_percentage": None,
        }
    )

    assert cfg == {
        "dataset_with_size_type": "Small Cohere (768dim, 100K)",
        "tenant_count": 13,
        "tenant_prefix": "acct_",
        "tenant_id_width": 3,
        "payload_profile": "vector",
        "filter_rate": 0.01,
    }


def test_cli_omits_multitenant_dataset_when_not_selected():
    cfg = get_custom_case_config(
        {
            "case_type": "CloudMultiTenantSearchCase",
            "dataset_with_size_type": None,
            "tenant_count": 13,
            "tenant_prefix": "acct_",
            "tenant_id_width": 3,
            "payload_profile": "ids_only",
            "cloud_filter_rate": None,
            "cloud_label_percentage": None,
        }
    )

    assert cfg == {
        "tenant_count": 13,
        "tenant_prefix": "acct_",
        "tenant_id_width": 3,
        "payload_profile": "ids_only",
    }

    case = CaseConfig(case_id=CaseType.CloudMultiTenantSearchCase, custom_case=cfg).case
    assert case.dataset_with_size_type == DatasetWithSizeType.CohereLarge
    assert case.dataset.data.size == 10_000_000


def test_assembler_schedules_cloud_insert_case():
    task = TaskConfig(
        db=DB.ZillizCloud,
        db_config=ZillizCloudConfig(uri="https://example.com", user="db_admin"),
        db_case_config=AutoIndexConfig(),
        case_config=CaseConfig(
            case_id=CaseType.CloudInsertCase,
            custom_case={
                "batch_size": 1000,
                "dataset_with_size_type": DatasetWithSizeType.CohereMedium.value,
            },
        ),
        stages=[TaskStage.DROP_OLD, TaskStage.LOAD],
    )

    runner = Assembler.assemble_all("run-id", "task-label", [task], DatasetSource.S3)

    assert len(runner.case_runners) == 1
    assert runner.case_runners[0].ca.label == CaseLabel.CloudInsert


def test_default_insert_readiness_is_immediately_ready():
    class FakeVectorDB(VectorDB):
        def __init__(self, dim, db_config, db_case_config, collection_name="", drop_old=False):
            pass

        @contextmanager
        def init(self):
            yield

        def insert_embeddings(self, embeddings, metadata, labels_data=None, **kwargs):
            return len(embeddings), None

        def search_embedding(self, query, k=100, payload_profile=None):
            return []

        def optimize(self, data_size=None):
            pass

    assert FakeVectorDB(1, {}, EmptyDBCaseConfig()).poll_insert_readiness(10) == {
        "fully_searchable": True,
        "fully_indexed": True,
        "additional_parameters": {},
    }


def test_metric_contains_cloud_insert_output_fields():
    metric = Metric(
        inserted_count=100,
        insert_rows_per_second=83.33,
        insert_completion_seconds=1.2,
        searchable_after_insert_seconds=3.4,
        indexed_after_searchable_seconds=5.6,
        additional_parameters={"disable_backpressure": True},
    )

    assert metric.inserted_count == 100
    assert metric.insert_rows_per_second == 83.33
    assert metric.insert_completion_seconds == 1.2
    assert metric.searchable_after_insert_seconds == 3.4
    assert metric.indexed_after_searchable_seconds == 5.6
    assert metric.additional_parameters == {"disable_backpressure": True}


def test_cloud_insert_result_file_uses_insert_only_metrics(tmp_path: Path):
    result = CaseResult(
        task_config=TaskConfig(
            db=DB.Pinecone,
            db_config=PineconeConfig(
                db_label="pinecone_cloud_insert_laion100m_bs1k",
                api_key="secret-key",
                index_name="laion100m",
            ),
            db_case_config=EmptyDBCaseConfig(),
            case_config=CaseConfig(
                case_id=CaseType.CloudInsertCase,
                custom_case={"batch_size": 1000, "duration": None},
            ),
            stages=[TaskStage.DROP_OLD, TaskStage.LOAD],
            load_concurrency=0,
        ),
        metrics=Metric(
            inserted_count=100_000_000,
            insert_rows_per_second=3919.9296,
            insert_completion_seconds=255.1066,
            searchable_after_insert_seconds=0.0,
            indexed_after_searchable_seconds=28.5956,
            additional_parameters={},
        ),
    )
    test_result = TestResult(run_id="run-id", task_label="cloud_insert_pinecone_laion100m_bs1k", results=[result])

    test_result.write_db_file(tmp_path, test_result, "pinecone")

    result_file = next(tmp_path.glob("result_*_pinecone.json"))
    raw_output = result_file.read_text()
    assert raw_output.startswith('{\n  "run_id"')
    written = json.loads(raw_output)
    assert written["results"][0]["metrics"] == {
        "inserted_count": 100_000_000,
        "insert_rows_per_second": 3919.9296,
        "insert_completion_seconds": 255.1066,
        "searchable_after_insert_seconds": 0.0,
        "indexed_after_searchable_seconds": 28.5956,
        "additional_parameters": {},
    }
    assert written["results"][0]["task_config"]["db_config"]["api_key"] == "**********"
    assert written["results"][0]["task_config"]["db_config"]["index_name"] == "laion100m"
    assert written["results"][0]["task_config"]["case_config"] == {
        "case_id": 600,
        "custom_case": {"batch_size": 1000, "duration": None},
    }

    read_back = TestResult.read_file(result_file)
    assert read_back.results[0].task_config.case_config.case_id == CaseType.CloudInsertCase
    assert read_back.results[0].task_config.case_config.custom_case == {"batch_size": 1000, "duration": None}


def test_turbopuffer_insert_can_disable_backpressure():
    db = TurboPuffer.__new__(TurboPuffer)
    db.with_scalar_labels = False
    db._scalar_id_field = "id"
    db._vector_field = "vector"
    db.metric = "cosine_distance"
    db.db_case_config = TurboPufferIndexConfig(disable_backpressure=True)

    class Namespace:
        kwargs = None

        def write(self, **kwargs):
            self.kwargs = kwargs

    db.ns = Namespace()

    assert db.insert_embeddings([[0.1]], [1]) == (1, None)
    assert db.ns.kwargs["disable_backpressure"] is True


def test_turbopuffer_insert_serializes_numpy_vectors():
    db = TurboPuffer.__new__(TurboPuffer)
    db.with_scalar_labels = False
    db._scalar_id_field = "id"
    db._vector_field = "vector"
    db.metric = "cosine_distance"
    db.db_case_config = TurboPufferIndexConfig(disable_backpressure=False)

    class Namespace:
        kwargs = None

        def write(self, **kwargs):
            self.kwargs = kwargs

    db.ns = Namespace()

    insert_count, error = db.insert_embeddings([np.array([0.1, 0.2])], [1])

    assert error is None
    assert insert_count == 1
    assert db.ns.kwargs["upsert_columns"]["vector"] == [[0.1, 0.2]]


def test_turbopuffer_insert_returns_write_error():
    db = TurboPuffer.__new__(TurboPuffer)
    db.with_scalar_labels = False
    db._scalar_id_field = "id"
    db._vector_field = "vector"
    db.metric = "cosine_distance"
    db.db_case_config = TurboPufferIndexConfig(disable_backpressure=False)

    class Namespace:
        def write(self, **kwargs):
            raise RuntimeError("write failed")

    db.ns = Namespace()

    insert_count, error = db.insert_embeddings([[0.1]], [1])

    assert insert_count == 0
    assert isinstance(error, RuntimeError)


def test_milvus_insert_readiness_uses_entity_count_and_index_progress():
    db = Milvus.__new__(Milvus)
    db.collection_name = "c"
    db._vector_index_name = "vector_idx"
    db.client = type(
        "Client",
        (),
        {
            "flush": lambda self, collection_name: None,
            "get_collection_stats": lambda self, collection_name: {"row_count": "10"},
            "describe_index": lambda self, collection_name, index_name: {"pending_index_rows": 0},
        },
    )()

    assert db.poll_insert_readiness(10) == {
        "fully_searchable": True,
        "fully_indexed": True,
        "additional_parameters": {},
    }


def test_pinecone_insert_readiness_uses_vector_count():
    db = Pinecone.__new__(Pinecone)
    db.index = type("Index", (), {"describe_index_stats": lambda self: {"total_vector_count": 9}})()

    assert db.poll_insert_readiness(10)["fully_searchable"] is False
    assert db.poll_insert_readiness(9)["fully_indexed"] is True


def test_pinecone_supports_cloud_payload_profiles():
    db = Pinecone.__new__(Pinecone)

    assert db.supports_payload_profile(PayloadProfile.IDS_ONLY)
    assert db.supports_payload_profile(PayloadProfile.SCALAR_LABEL)
    assert db.supports_payload_profile(PayloadProfile.VECTOR)


def test_pinecone_search_requests_metadata_for_scalar_label_payload():
    db = Pinecone.__new__(Pinecone)
    db.expr = None

    class Index:
        kwargs = None

        def query(self, **kwargs):
            self.kwargs = kwargs
            return {"matches": [{"id": "1"}]}

    db.index = Index()

    assert db.search_embedding([0.1], payload_profile=PayloadProfile.SCALAR_LABEL) == [1]
    assert db.index.kwargs["include_metadata"] is True
    assert db.index.kwargs["include_values"] is False


def test_pinecone_search_requests_values_for_vector_payload():
    db = Pinecone.__new__(Pinecone)
    db.expr = {"meta": {"$gte": 10}}

    class Index:
        kwargs = None

        def query(self, **kwargs):
            self.kwargs = kwargs
            return {"matches": [{"id": "2"}]}

    db.index = Index()

    assert db.search_embedding([0.1], payload_profile=PayloadProfile.VECTOR) == [2]
    assert db.index.kwargs["include_metadata"] is False
    assert db.index.kwargs["include_values"] is True
    assert db.index.kwargs["filter"] == {"meta": {"$gte": 10}}


def test_pinecone_search_retries_rate_limited_queries(monkeypatch):
    db = Pinecone.__new__(Pinecone)
    db.expr = None
    monkeypatch.setenv("PINECONE_QUERY_RETRY_SLEEP_SECONDS", "0.25")
    sleeps = []
    monkeypatch.setattr("vectordb_bench.backend.clients.pinecone.pinecone.time.sleep", sleeps.append)

    class RateLimitError(Exception):
        status = 429

    class Index:
        calls = 0

        def query(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RateLimitError("too many requests")
            return {"matches": [{"id": "3"}]}

    db.index = Index()

    assert db.search_embedding([0.1]) == [3]
    assert db.index.calls == 3
    assert sleeps == [0.25, 0.25]


def test_pinecone_search_stops_after_rate_limit_retry_budget(monkeypatch):
    db = Pinecone.__new__(Pinecone)
    db.expr = None
    monkeypatch.setenv("PINECONE_QUERY_MAX_RETRIES", "1")
    monkeypatch.setattr("vectordb_bench.backend.clients.pinecone.pinecone.time.sleep", lambda _: None)

    class RateLimitError(Exception):
        status = 429

    class Index:
        def query(self, **kwargs):
            raise RateLimitError("too many requests")

    db.index = Index()

    try:
        db.search_embedding([0.1])
    except RateLimitError:
        pass
    else:
        raise AssertionError("expected Pinecone rate limit error")


def test_pinecone_insert_tracks_last_write_lsn():
    db = Pinecone.__new__(Pinecone)
    db.batch_size = 1000
    db.with_scalar_labels = False
    db._scalar_id_field = "meta"

    class UpsertResponse:
        _response_info = {"raw_headers": {"x-pinecone-request-lsn": "123"}}

    class Index:
        def upsert(self, records):
            return UpsertResponse()

    db.index = Index()

    insert_count, error = db.insert_embeddings([[0.1]], [1])

    assert error is None
    assert insert_count == 1
    assert db._last_write_lsn == 123


def test_pinecone_insert_keeps_highest_write_lsn():
    db = Pinecone.__new__(Pinecone)
    db.batch_size = 1
    db.with_scalar_labels = False
    db._scalar_id_field = "meta"
    responses = iter(["123", "122"])

    class UpsertResponse:
        def __init__(self, lsn):
            self._response_info = {"raw_headers": {"x-pinecone-request-lsn": lsn}}

    class Index:
        def upsert(self, records):
            return UpsertResponse(next(responses))

    db.index = Index()

    insert_count, error = db.insert_embeddings([[0.1], [0.2]], [1, 2])

    assert error is None
    assert insert_count == 2
    assert db._last_write_lsn == 123


def test_pinecone_record_write_lsn_keeps_highest_value():
    db = Pinecone.__new__(Pinecone)

    db._record_write_lsn(123)
    db._record_write_lsn(122)

    assert db._last_write_lsn == 123


def test_pinecone_insert_readiness_uses_lsn_when_available():
    db = Pinecone.__new__(Pinecone)
    db._last_write_lsn = 123
    db._readiness_probe_vector = [0.0]

    class QueryResponse(dict):
        def __init__(self, indexed_lsn):
            super().__init__({"matches": []})
            self._response_info = {"raw_headers": {"x-pinecone-max-indexed-lsn": str(indexed_lsn)}}

    class Index:
        indexed_lsn = 122

        def describe_index_stats(self):
            return {"total_vector_count": 10}

        def query(self, **kwargs):
            return QueryResponse(self.indexed_lsn)

    db.index = Index()

    assert db.poll_insert_readiness(10)["fully_searchable"] is False
    db.index.indexed_lsn = 123
    assert db.poll_insert_readiness(10)["fully_indexed"] is True


def test_turbopuffer_insert_readiness_uses_unindexed_bytes():
    db = TurboPuffer.__new__(TurboPuffer)
    db.db_case_config = TurboPufferIndexConfig(disable_backpressure=True)
    db.ns = type("Namespace", (), {"metadata": lambda self: {"unindexed_bytes": 1}})()

    status = db.poll_insert_readiness(10)

    assert status["fully_searchable"] is True
    assert status["fully_indexed"] is False
    assert status["additional_parameters"] == {"disable_backpressure": True}


def test_turbopuffer_insert_readiness_uses_nested_unindexed_bytes():
    db = TurboPuffer.__new__(TurboPuffer)
    db.db_case_config = TurboPufferIndexConfig(disable_backpressure=False)
    db.ns = type(
        "Namespace",
        (),
        {"metadata": lambda self: {"index": {"status": "updating", "unindexed_bytes": 1}}},
    )()

    status = db.poll_insert_readiness(10)

    assert status["fully_searchable"] is True
    assert status["fully_indexed"] is False
    assert status["additional_parameters"] == {"disable_backpressure": False}


def test_cloud_insert_runner_records_insert_and_readiness_metrics(monkeypatch):
    class Data:
        train_id_field = "id"
        train_vector_field = "vector"
        metric_type = "L2"

    class Dataset:
        data = Data()

        def iter_batches(self, batch_size):
            assert batch_size == 2
            return iter(
                [
                    pd.DataFrame({"id": [1, 2], "vector": [np.array([0.1]), np.array([0.2])]}),
                    pd.DataFrame({"id": [3], "vector": [np.array([0.3])]}),
                ]
            )

    class DB:
        thread_safe = True
        name = "fake"
        inserts = []
        readiness_calls = 0

        @contextmanager
        def init(self):
            yield

        def need_normalize_cosine(self):
            return False

        def insert_embeddings(self, embeddings, metadata, labels_data=None):
            self.inserts.append((embeddings, metadata))
            return len(metadata), None

        def poll_insert_readiness(self, expected_count):
            self.readiness_calls += 1
            return {
                "fully_searchable": self.readiness_calls >= 2,
                "fully_indexed": self.readiness_calls >= 3,
                "additional_parameters": {"example": "value"},
            }

    db = DB()
    monkeypatch.setattr("vectordb_bench.backend.task_runner.time.sleep", lambda _: None)
    case = CloudInsertCase(batch_size=2)
    case.dataset = Dataset()
    config = type("Config", (), {"load_concurrency": 1})()
    runner = CaseRunner.construct(ca=case, db=db, config=config)

    metric = runner._run_cloud_insert_case()

    assert [metadata for _, metadata in db.inserts] == [[1, 2], [3]]
    assert metric.inserted_count == 3
    assert metric.insert_rows_per_second > 0
    assert metric.insert_completion_seconds >= 0
    assert metric.searchable_after_insert_seconds >= 0
    assert metric.indexed_after_searchable_seconds >= 0
    assert metric.additional_parameters == {"example": "value"}


def test_cloud_insert_runner_uses_concurrent_insert_runner(monkeypatch):
    created = {}

    class Data:
        metric_type = "L2"

    class Dataset:
        data = Data()

    class FakeConcurrentInsertRunner:
        def __init__(self, db, dataset, normalize, filters, max_workers, batch_size, duration):
            created.update(
                {
                    "db": db,
                    "dataset": dataset,
                    "normalize": normalize,
                    "filters": filters,
                    "max_workers": max_workers,
                    "batch_size": batch_size,
                    "duration": duration,
                }
            )

        def task(self):
            return 3

    class DB:
        @contextmanager
        def init(self):
            yield

        def need_normalize_cosine(self):
            return False

        def poll_insert_readiness(self, expected_count):
            return {"fully_searchable": True, "fully_indexed": True, "additional_parameters": {}}

    monkeypatch.setattr("vectordb_bench.backend.task_runner.ConcurrentInsertRunner", FakeConcurrentInsertRunner)
    case = CloudInsertCase(batch_size=1000, duration=60)
    case.dataset = Dataset()
    config = type("Config", (), {"load_concurrency": 7})()
    runner = CaseRunner.construct(ca=case, db=DB(), config=config)

    metric = runner._run_cloud_insert_case()

    assert metric.inserted_count == 3
    assert created["batch_size"] == 1000
    assert created["duration"] == 60
    assert created["max_workers"] == 7
    assert created["dataset"] is case.dataset


def test_concurrent_insert_runner_uses_custom_batch_size_iterator():
    class Data:
        train_id_field = "id"
        train_vector_field = "vector"

    class Dataset:
        data = Data()
        requested_batch_size = None

        def iter_batches(self, batch_size):
            self.requested_batch_size = batch_size
            return iter(
                [
                    pd.DataFrame(
                        {
                            "id": [1, 2],
                            "vector": [np.array([0.1]), np.array([0.2])],
                        }
                    )
                ]
            )

        def __iter__(self):
            raise AssertionError("ConcurrentInsertRunner should request an explicit batch size")

    class DB:
        thread_safe = True
        name = "fake"

        def __init__(self):
            self.inserts = []

        @contextmanager
        def init(self):
            yield

        def insert_embeddings(self, embeddings, metadata, labels_data=None):
            self.inserts.append((embeddings, metadata))
            return len(metadata), None

    dataset = Dataset()
    db = DB()
    runner = ConcurrentInsertRunner(db, dataset, normalize=False, max_workers=1, batch_size=1000)

    assert runner.task() == 2
    assert dataset.requested_batch_size == 1000
    assert db.inserts == [([[0.1], [0.2]], [1, 2])]


class TenantInsertProbeDB:
    name = "TenantInsertProbeDB"
    thread_safe = True

    def __init__(self):
        self.calls = []

    @contextmanager
    def init(self):
        yield

    def insert_embeddings(self, embeddings, metadata, labels_data=None, tenant_labels_data=None):
        self.calls.append(
            {
                "embeddings": embeddings,
                "metadata": metadata,
                "labels_data": labels_data,
                "tenant_labels_data": tenant_labels_data,
            }
        )
        return len(embeddings), None


class TenantAwareCase:
    is_multitenant = True

    def tenant_labels_for_ids(self, row_ids):
        return [f"tenant_{int(row_id) % 3:04d}" for row_id in row_ids]


def test_concurrent_insert_runner_passes_tenant_labels():
    db = TenantInsertProbeDB()
    dataset = MagicMock()
    dataset.data.train_id_field = "id"
    dataset.data.train_vector_field = "emb"
    dataset.iter_batches.return_value = iter(
        [
            pd.DataFrame(
                {
                    "id": [0, 1, 5],
                    "emb": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                }
            )
        ]
    )

    runner = ConcurrentInsertRunner(
        db=db,
        dataset=dataset,
        normalize=False,
        max_workers=1,
        batch_size=10,
        tenant_case=TenantAwareCase(),
    )

    count = runner.task()

    assert count == 3
    assert db.calls[0]["metadata"] == [0, 1, 5]
    assert db.calls[0]["tenant_labels_data"] == ["tenant_0000", "tenant_0001", "tenant_0002"]


def test_concurrent_insert_runner_passes_scalar_labels_for_scalar_payload_without_filter():
    db = TenantInsertProbeDB()
    dataset = MagicMock()
    dataset.data.train_id_field = "id"
    dataset.data.train_vector_field = "emb"
    dataset.data.scalar_labels_file_separated = False
    dataset.iter_batches.return_value = iter(
        [
            pd.DataFrame(
                {
                    "id": [0, 1, 5],
                    "emb": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                    "labels": ["label_a", "label_b", "label_c"],
                }
            )
        ]
    )

    runner = ConcurrentInsertRunner(
        db=db,
        dataset=dataset,
        normalize=False,
        max_workers=1,
        batch_size=10,
        with_scalar_labels=True,
    )

    count = runner.task()

    assert count == 3
    assert db.calls[0]["metadata"] == [0, 1, 5]
    assert db.calls[0]["labels_data"] == ["label_a", "label_b", "label_c"]
    assert db.calls[0]["tenant_labels_data"] is None
