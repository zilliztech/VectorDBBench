"""E2E test for Milvus client using MilvusClient API.

Requires a running Milvus instance at localhost:19530.
"""

import logging

from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.milvus.config import MilvusConfig
from vectordb_bench.backend.cases import CaseType
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
