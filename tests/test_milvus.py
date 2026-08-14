"""E2E test for Milvus client using MilvusClient API.

Requires a running Milvus instance at localhost:19530.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from pydantic import SecretStr

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.milvus.config import MilvusConfig
from vectordb_bench.backend.clients.milvus.milvus import MILVUS_FORCE_MERGE_TARGET_SIZE_MB, Milvus
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import CaseConfig, TaskConfig

log = logging.getLogger(__name__)


def test_milvus_vector_payload_requests_vector_field_and_returns_ids():
    captured = {}

    def search(**kwargs):
        captured.update(kwargs)
        return [[{"pk": 1, "vector": [0.1, 0.2]}]]

    db = object.__new__(Milvus)
    db.client = SimpleNamespace(search=search)
    db.collection_name = "test_collection"
    db._vector_field = "vector"
    db._primary_field = "pk"
    db._scalar_label_field = "label"
    db.case_config = SimpleNamespace(search_param=lambda: {"metric_type": "COSINE"})
    db.expr = ""

    result = db.search_embedding([0.1, 0.2], k=3, payload_profile=PayloadProfile.VECTOR)

    assert result == [1]
    assert captured["output_fields"] == ["vector"]


def _fake_milvus_client(monkeypatch, *, collection_exists=False, properties=None):
    client = MagicMock()
    client.has_collection.return_value = collection_exists
    client.describe_collection.return_value = {"properties": properties or {}}
    client_cls = MagicMock(return_value=client)
    client_cls.create_schema.return_value = MagicMock()
    client_cls.prepare_index_params.return_value = MagicMock()
    monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.MilvusClient", client_cls)
    return client


def _create_milvus_with_collection_properties(monkeypatch, *, collection_exists=False, properties=None):
    client = _fake_milvus_client(
        monkeypatch,
        collection_exists=collection_exists,
        properties=properties,
    )
    Milvus(
        dim=2,
        db_config={"uri": "http://example.invalid"},
        db_case_config=SimpleNamespace(
            index_param=lambda: {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        ),
        collection_properties={"query_mode": "large_topk"},
    )
    return client


def test_milvus_creates_collection_properties_before_index(monkeypatch):
    client = _create_milvus_with_collection_properties(monkeypatch)

    assert client.create_collection.call_args.kwargs["properties"] == {"query_mode": "large_topk"}
    method_names = [method_call[0] for method_call in client.method_calls]
    assert method_names.index("create_collection") < method_names.index("create_index")


def test_milvus_rejects_existing_collection_with_incompatible_properties(monkeypatch):
    with pytest.raises(ValueError, match="incompatible collection properties"):
        _create_milvus_with_collection_properties(monkeypatch, collection_exists=True)


class TestMilvusOptimize:
    def _milvus(
        self,
        *,
        compact_side_effect: Exception | None = None,
        is_fts: bool = False,
        is_gpu_index: bool = False,
    ):
        milvus = Milvus.__new__(Milvus)
        milvus.name = "Milvus"
        milvus.collection_name = "test_collection"
        milvus._is_fts = is_fts
        milvus.case_config = SimpleNamespace(is_gpu_index=is_gpu_index)
        milvus.client = MagicMock()
        milvus.client.compact.side_effect = compact_side_effect
        milvus.client.compact.return_value = 42
        milvus._wait_for_segments_sorted = MagicMock()
        milvus._wait_for_index = MagicMock()
        milvus._wait_for_compaction = MagicMock()
        return milvus

    def test_optimize_compact_uses_safe_force_merge_target_size(self):
        milvus = self._milvus()

        milvus._optimize()

        milvus.client.compact.assert_any_call("test_collection", target_size=MILVUS_FORCE_MERGE_TARGET_SIZE_MB)
        milvus.client.refresh_load.assert_called_once_with("test_collection")

    def test_optimize_compacts_fts_collections(self):
        milvus = self._milvus(is_fts=True)

        milvus._optimize()

        milvus.client.compact.assert_any_call("test_collection", target_size=MILVUS_FORCE_MERGE_TARGET_SIZE_MB)
        milvus.client.refresh_load.assert_called_once_with("test_collection")

    def test_optimize_flushes_and_runs_normal_compaction_before_force_merge(self):
        milvus = self._milvus(compact_side_effect=[41, 42])

        milvus._optimize()

        milvus.client.flush.assert_called_once_with("test_collection")
        assert milvus.client.compact.call_args_list == [
            call("test_collection"),
            call("test_collection", target_size=MILVUS_FORCE_MERGE_TARGET_SIZE_MB),
        ]
        assert milvus._wait_for_segments_sorted.call_count == 2
        assert milvus._wait_for_index.call_count == 3
        assert milvus._wait_for_compaction.call_args_list == [call(41), call(42)]
        milvus.client.refresh_load.assert_called_once_with("test_collection")

    def test_optimize_retries_when_compacting_segments_are_missing_from_force_merge_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        class FakeMilvusClient:
            def __init__(self):
                self.segment_ids = [1, 2, 3]
                self.force_merge_completed = False

            def flush(self, _collection_name: str):
                pass

            def list_persistent_segments(self, _collection_name: str):
                return [
                    SimpleNamespace(
                        segment_id=segment_id,
                        is_sorted=True,
                        state_name="Flushed",
                        level_name="L1",
                    )
                    for segment_id in self.segment_ids
                ]

            def describe_index(self, _collection_name: str, _index_name: str):
                return {"pending_index_rows": 0}

            def compact(self, _collection_name: str, *, target_size: int | None = None):
                if target_size is None:
                    return 90
                assert target_size == MILVUS_FORCE_MERGE_TARGET_SIZE_MB
                if self.segment_ids == [1, 2, 3]:
                    return 100
                if self.segment_ids == [10, 20]:
                    return 101
                message = f"unexpected force merge input: {self.segment_ids}"
                raise AssertionError(message)

            def get_compaction_state(self, compaction_id: int):
                if compaction_id == 90:
                    pass
                elif compaction_id == 100:
                    self.segment_ids = [10, 20]
                elif compaction_id == 101:
                    self.segment_ids = [30]
                    self.force_merge_completed = True
                else:
                    message = f"unexpected compaction id: {compaction_id}"
                    raise AssertionError(message)
                return "Completed"

            def get_compaction_plans(self, compaction_id: int):
                sources = [1] if compaction_id == 100 else [10, 20]
                return SimpleNamespace(plans=[SimpleNamespace(sources=sources)])

            def refresh_load(self, _collection_name: str):
                assert self.force_merge_completed, "refresh started after only a partial force merge"

        milvus = Milvus.__new__(Milvus)
        milvus.name = "Milvus"
        milvus.collection_name = "test_collection"
        milvus._is_fts = False
        milvus._main_index_name = "vector_idx"
        milvus.case_config = SimpleNamespace(is_gpu_index=False)
        milvus.client = FakeMilvusClient()
        monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.time.sleep", lambda _seconds: None)

        milvus.optimize(data_size=500_000)

        assert milvus.client.force_merge_completed

    def test_optimize_retries_when_all_segments_are_compacting(self, monkeypatch: pytest.MonkeyPatch):
        class FakeMilvusClient:
            def __init__(self):
                self.segment_ids = [1, 2]
                self.compact_attempts = 0
                self.force_merge_completed = False

            def flush(self, _collection_name: str):
                pass

            def list_persistent_segments(self, _collection_name: str):
                return [
                    SimpleNamespace(
                        segment_id=segment_id,
                        is_sorted=True,
                        state_name="Flushed",
                        level_name="L1",
                    )
                    for segment_id in self.segment_ids
                ]

            def describe_index(self, _collection_name: str, _index_name: str):
                return {"pending_index_rows": 0}

            def compact(self, _collection_name: str, *, target_size: int | None = None):
                if target_size is None:
                    return 90
                assert target_size == MILVUS_FORCE_MERGE_TARGET_SIZE_MB
                self.compact_attempts += 1
                if self.compact_attempts == 1:
                    self.segment_ids = [10]
                    return -1
                return 101

            def get_compaction_state(self, compaction_id: int):
                if compaction_id == 90:
                    return "Completed"
                assert compaction_id == 101
                self.segment_ids = [20]
                self.force_merge_completed = True
                return "Completed"

            def get_compaction_plans(self, compaction_id: int):
                assert compaction_id == 101
                return SimpleNamespace(plans=[SimpleNamespace(sources=[10])])

            def refresh_load(self, _collection_name: str):
                assert self.force_merge_completed, "refresh started without a force merge job"

        milvus = Milvus.__new__(Milvus)
        milvus.name = "Milvus"
        milvus.collection_name = "test_collection"
        milvus._is_fts = False
        milvus._main_index_name = "vector_idx"
        milvus.case_config = SimpleNamespace(is_gpu_index=False)
        milvus.client = FakeMilvusClient()
        monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.time.sleep", lambda _seconds: None)

        milvus.optimize(data_size=500_000)

        assert milvus.client.force_merge_completed

    def test_force_merge_uses_fresh_snapshots_and_stops_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch):
        class FakeMilvusClient:
            def __init__(self):
                self.segment_ids = [1, 2]
                self.compact_attempts = 0

            def list_persistent_segments(self, _collection_name: str):
                return [
                    SimpleNamespace(
                        segment_id=segment_id,
                        is_sorted=True,
                        state_name="Flushed",
                        level_name="L1",
                    )
                    for segment_id in self.segment_ids
                ]

            def describe_index(self, _collection_name: str, _index_name: str):
                return {"pending_index_rows": 0}

            def compact(self, _collection_name: str, *, target_size: int):
                assert target_size == MILVUS_FORCE_MERGE_TARGET_SIZE_MB
                self.compact_attempts += 1
                return 100 + self.compact_attempts

            def get_compaction_state(self, _compaction_id: int):
                next_segment_ids = {
                    1: [10, 20],
                    2: [30, 40],
                    3: [50, 60],
                }
                self.segment_ids = next_segment_ids[self.compact_attempts]
                return "Completed"

            def get_compaction_plans(self, _compaction_id: int):
                planned_source = {1: 1, 2: 10, 3: 30}[self.compact_attempts]
                return SimpleNamespace(plans=[SimpleNamespace(sources=[planned_source])])

        milvus = Milvus.__new__(Milvus)
        milvus.name = "Milvus"
        milvus.collection_name = "test_collection"
        milvus._main_index_name = "vector_idx"
        milvus.client = FakeMilvusClient()
        monkeypatch.setattr("vectordb_bench.backend.clients.milvus.milvus.time.sleep", lambda _seconds: None)

        with pytest.raises(RuntimeError, match=r"after 3 attempts.*missing segments: \[40\]"):
            milvus._force_merge(max_attempts=3)

        assert milvus.client.compact_attempts == 3

    def test_optimize_skips_gpu_index_compaction(self):
        milvus = self._milvus(is_gpu_index=True)

        milvus._optimize()

        milvus.client.compact.assert_not_called()
        milvus.client.refresh_load.assert_called_once_with("test_collection")

    def test_optimize_skips_property_style_permission_denied(self):
        error = RuntimeError("permission denied")
        error.code = SimpleNamespace(name="PERMISSION_DENIED")
        milvus = self._milvus(compact_side_effect=error)

        milvus._optimize()

        milvus.client.refresh_load.assert_called_once_with("test_collection")

    def test_optimize_reraises_non_permission_error(self):
        error = RuntimeError("boom")
        error.code = SimpleNamespace(name="UNAVAILABLE")
        milvus = self._milvus(compact_side_effect=error)

        with pytest.raises(RuntimeError, match="boom") as exc_info:
            milvus._optimize()

        assert exc_info.value is error
        milvus.client.refresh_load.assert_not_called()


@pytest.mark.integration
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


def test_milvus_multitenant_insert_writes_tenant_and_scalar_payload_labels() -> None:
    inserted = {}

    def insert(collection_name, batch_data):
        inserted["collection_name"] = collection_name
        inserted["batch_data"] = batch_data
        return {"insert_count": len(batch_data)}

    db = object.__new__(Milvus)
    db.client = SimpleNamespace(insert=insert)
    db.collection_name = "test_collection"
    db.batch_size = 100
    db._primary_field = "pk"
    db._scalar_id_field = "id"
    db._vector_field = "vector"
    db._scalar_label_field = "label"
    db._scalar_payload_label_field = "scalar_label"
    db._multitenant_partition_key_field = "labels"
    db.with_scalar_labels = True

    count, err = db.insert_embeddings(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadata=[1, 2],
        labels_data=["label_a", "label_b"],
        tenant_labels_data=["tenant_0001", "tenant_0002"],
    )

    assert count == 2
    assert err is None
    assert inserted["batch_data"] == [
        {"pk": 1, "id": 1, "vector": [0.1, 0.2], "labels": "tenant_0001", "scalar_label": "label_a"},
        {"pk": 2, "id": 2, "vector": [0.3, 0.4], "labels": "tenant_0002", "scalar_label": "label_b"},
    ]
