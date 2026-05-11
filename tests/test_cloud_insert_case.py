from contextlib import contextmanager

import polars as pl

from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.backend.cases import CaseLabel, CaseType, CloudInsertCase
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig, VectorDB
from vectordb_bench.backend.clients.milvus import milvus as milvus_module
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.clients.pinecone.pinecone import Pinecone
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig


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
        custom_case={"batch_size": 5000, "duration": 1800},
    ).case

    assert isinstance(case, CloudInsertCase)
    assert case.batch_size == 5000
    assert case.duration == 1800


def test_cli_builds_cloud_insert_custom_case_config():
    params = {
        "case_type": "CloudInsertCase",
        "cloud_insert_batch_size": 10_000,
        "cloud_insert_duration": 1800,
    }

    assert get_custom_case_config(params) == {
        "batch_size": 10_000,
        "duration": 1800,
    }


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
        insert_completion_seconds=1.2,
        searchable_after_insert_seconds=3.4,
        indexed_after_searchable_seconds=5.6,
        additional_parameters={"disable_backpressure": True},
    )

    assert metric.insert_completion_seconds == 1.2
    assert metric.searchable_after_insert_seconds == 3.4
    assert metric.indexed_after_searchable_seconds == 5.6
    assert metric.additional_parameters == {"disable_backpressure": True}


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


def test_milvus_insert_readiness_uses_entity_count_and_index_progress(monkeypatch):
    class Collection:
        num_entities = 10

        def flush(self):
            pass

    db = Milvus.__new__(Milvus)
    db.col = Collection()
    db.collection_name = "c"
    db._vector_index_name = "vector_idx"
    monkeypatch.setattr(
        milvus_module.utility,
        "index_building_progress",
        lambda collection_name, index_name: {"pending_index_rows": 0},
    )

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


def test_turbopuffer_insert_readiness_uses_unindexed_bytes():
    db = TurboPuffer.__new__(TurboPuffer)
    db.db_case_config = TurboPufferIndexConfig(disable_backpressure=True)
    db.ns = type("Namespace", (), {"metadata": lambda self: {"unindexed_bytes": 1}})()

    status = db.poll_insert_readiness(10)

    assert status["fully_searchable"] is True
    assert status["fully_indexed"] is False
    assert status["additional_parameters"] == {"disable_backpressure": True}


def test_cloud_insert_runner_records_insert_and_readiness_metrics(monkeypatch):
    class Data:
        train_id_field = "id"
        train_vector_field = "vector"
        metric_type = "L2"

    class Dataset:
        data = Data()

        def __iter__(self):
            yield pl.DataFrame({"id": [1, 2, 3], "vector": [[0.1], [0.2], [0.3]]})

    class DB:
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
    runner = CaseRunner.construct(ca=case, db=db)

    metric = runner._run_cloud_insert_case()

    assert [metadata for _, metadata in db.inserts] == [[1, 2], [3]]
    assert metric.insert_completion_seconds >= 0
    assert metric.searchable_after_insert_seconds >= 0
    assert metric.indexed_after_searchable_seconds >= 0
    assert metric.additional_parameters == {"example": "value"}
