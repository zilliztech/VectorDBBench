"""E2E test for Milvus client using MilvusClient API.

Requires a running Milvus instance at localhost:19530.
"""

import logging
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.milvus.config import MilvusConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import CaseConfig, TaskConfig

log = logging.getLogger(__name__)


class TestMilvus:
    """E2E test for Milvus using Performance1536D50K (OpenAI 50K dataset)."""

    def test_performance_1536d_50k(self):
        """Full benchmark: download dataset, insert, optimize (force merge), search."""
        runner = BenchMarkRunner()

        task_config = TaskConfig(
            db=DB.Milvus,
            db_config=MilvusConfig(uri=SecretStr("http://localhost:19530")),
            db_case_config=DB.Milvus.case_config_cls(index_type=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.Performance1536D50K),
        )

        runner.run([task_config])
        runner._sync_running_task()
        result = runner.get_results()
        log.info(f"test result: {result}")
        assert len(result) > 0


def test_milvus_multitenant_search_uses_tenant_label_filter():
    captured = {}

    def search(**kwargs):
        captured.update(kwargs)
        return [[{"pk": 1}]]

    db = object.__new__(Milvus)
    db.client = SimpleNamespace(search=search)
    db.collection_name = "test_collection"
    db._vector_field = "vector"
    db._primary_field = "pk"
    db._scalar_label_field = "label"
    db.case_config = SimpleNamespace(search_param=lambda: {"metric_type": "COSINE"})
    db.expr = ""

    result = db.search_embedding([0.1, 0.2], k=3, payload_profile=PayloadProfile.IDS_ONLY, tenant="tenant_0003")

    assert result == [1]
    assert captured["filter"] == "label == 'tenant_0003'"


def test_milvus_validate_multitenant_schema_accepts_partition_key_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = []

    class FakeMilvusClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def describe_collection(self, _collection_name: str) -> dict:
            return {
                "fields": [
                    {"name": "pk", "is_primary": True},
                    {"name": "label", "is_partition_key": True},
                ]
            }

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.MilvusClient", FakeMilvusClient)

    db = object.__new__(Milvus)
    db.name = "Milvus"
    db.db_config = {"uri": "http://example.invalid", "user": None, "password": None, "token": ""}
    db.collection_name = "existing"
    db._scalar_label_field = "label"

    db.validate_multitenant_schema()

    assert closed == [True]


def test_milvus_validate_multitenant_schema_rejects_non_partition_key_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMilvusClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def describe_collection(self, _collection_name: str) -> dict:
            return {"fields": [{"name": "label", "is_partition_key": False}]}

        def close(self) -> None:
            pass

    monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.MilvusClient", FakeMilvusClient)

    db = object.__new__(Milvus)
    db.name = "Milvus"
    db.db_config = {"uri": "http://example.invalid", "user": None, "password": None, "token": ""}
    db.collection_name = "existing"
    db._scalar_label_field = "label"

    with pytest.raises(ValueError, match="label field is not a partition key"):
        db.validate_multitenant_schema()


def test_milvus_validate_multitenant_schema_uses_existing_labels_partition_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeMilvusClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def describe_collection(self, _collection_name: str) -> dict:
            return {
                "fields": [
                    {"name": "pk", "is_primary": True},
                    {"name": "labels", "is_partition_key": True},
                    {"name": "scalar_label", "nullable": True},
                ]
            }

        def close(self) -> None:
            pass

    def search(**kwargs):
        captured.update(kwargs)
        return [[{"pk": 1}]]

    monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.MilvusClient", FakeMilvusClient)

    db = object.__new__(Milvus)
    db.name = "Milvus"
    db.db_config = {"uri": "http://example.invalid", "user": None, "password": None, "token": ""}
    db.collection_name = "existing"
    db._vector_field = "vector"
    db._primary_field = "pk"
    db._scalar_label_field = "label"
    db.case_config = SimpleNamespace(search_param=lambda: {"metric_type": "COSINE"})
    db.expr = ""

    db.validate_multitenant_schema()
    db.client = SimpleNamespace(search=search)

    db.search_embedding([0.1, 0.2], payload_profile=PayloadProfile.SCALAR_LABEL, tenant="tenant_0003")

    assert captured["filter"] == "labels == 'tenant_0003'"
    assert captured["output_fields"] == ["scalar_label"]
