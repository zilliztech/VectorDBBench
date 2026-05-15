"""E2E test for Milvus client using MilvusClient API.

Requires a running Milvus instance at localhost:19530.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.milvus.config import MilvusConfig
from vectordb_bench.backend.clients.milvus.milvus import MILVUS_FORCE_MERGE_TARGET_SIZE_MB, Milvus
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import CaseConfig, TaskConfig

log = logging.getLogger(__name__)


class TestMilvusOptimize:
    def _milvus(self, *, compact_side_effect: Exception | None = None):
        milvus = Milvus.__new__(Milvus)
        milvus.name = "Milvus"
        milvus.collection_name = "test_collection"
        milvus.case_config = SimpleNamespace(is_gpu_index=False)
        milvus.client = MagicMock()
        milvus.client.compact.side_effect = compact_side_effect
        milvus.client.compact.return_value = 0
        milvus._wait_for_segments_sorted = MagicMock()
        milvus._wait_for_index = MagicMock()
        milvus._wait_for_compaction = MagicMock()
        return milvus

    def test_optimize_compact_uses_safe_force_merge_target_size(self):
        milvus = self._milvus()

        milvus._optimize()

        milvus.client.compact.assert_called_once_with("test_collection", target_size=MILVUS_FORCE_MERGE_TARGET_SIZE_MB)
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
