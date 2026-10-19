"""Tests for PgVector client and ConcurrentInsertRunner.

Reproduces issue #756: insert fails with
  TypeError: no default __reduce__ due to non-trivial __cinit__
when ConcurrentInsertRunner deep-copies a PgVector instance that has a live
psycopg connection open (the connection is opened by `with self.db.init():`
inside task() before the deepcopy in _get_thread_db()).

Requires:
  docker run -d --name pgvector-test \
    -e POSTGRES_USER=vectordb -e POSTGRES_PASSWORD=vectordb \
    -e POSTGRES_DB=vectordb -p 5432:5432 \
    pgvector/pgvector:pg17

Usage:
  pytest tests/test_pgvector.py -v -s
"""

from __future__ import annotations

import logging
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.pgvector.config import PgVectorHNSWConfig
from vectordb_bench.backend.dataset import Dataset, DatasetSource
from vectordb_bench.backend.filter import Filter, FilterOp, non_filter
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner

log = logging.getLogger(__name__)

# ── Connection config ────────────────────────────────────────────────────────

DB_CONFIG = {
    "connect_config": {
        "host": "localhost",
        "port": 5432,
        "dbname": "vectordb",
        "user": "vectordb",
        "password": "vectordb",
    },
    "table_name": "test_pgvector",
}

DIM = 128
COUNT = 500
RNG = np.random.default_rng(42)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_hnsw_config(**kwargs) -> PgVectorHNSWConfig:
    return PgVectorHNSWConfig(
        metric_type="COSINE",
        m=16,
        ef_construction=64,
        ef_search=64,
        **kwargs,
    )


def make_db(table_name: str = "test_pgvector", drop_old: bool = True) -> DB.PgVector.init_cls:
    cfg = dict(DB_CONFIG)
    cfg["table_name"] = table_name
    return DB.PgVector.init_cls(
        dim=DIM,
        db_config=cfg,
        db_case_config=make_hnsw_config(),
        drop_old=drop_old,
    )


def random_embeddings(n: int = COUNT, d: int = DIM) -> list[list[float]]:
    return RNG.random((n, d)).tolist()


# ── Basic client tests ────────────────────────────────────────────────────────


class TestPgVectorBasic:
    """Unit tests for the PgVector client (no subprocess)."""

    def test_insert_and_search(self):
        db = make_db("test_basic")
        embeddings = random_embeddings()
        metadata = list(range(COUNT))

        with db.init():
            count, err = db.insert_embeddings(embeddings=embeddings, metadata=metadata)
        assert err is None, f"Insert error: {err}"
        assert count == COUNT

        with db.init():
            db.optimize()

        with db.init():
            db.prepare_filter(Filter(type=FilterOp.NonFilter))
            results = db.search_embedding(query=embeddings[0], k=10)
        assert len(results) > 0

    def test_db_is_not_thread_safe(self):
        db = make_db("test_thread_safe")
        assert db.thread_safe is False

    def test_db_picklable_after_init(self):
        """PgVector instance must be picklable after __init__ (conn/cursor are None).

        This is required for ConcurrentInsertRunner which spawns a subprocess
        and pickles self (which includes self.db).
        """
        db = make_db("test_pickle")
        data = pickle.dumps(db)
        db2 = pickle.loads(data)  # noqa: S301
        assert db2.dim == DIM

    def test_get_thread_db_with_open_connection(self):
        """Regression test for issue #756.

        ConcurrentInsertRunner.task() opens `with self.db.init()` before calling
        workers. For non-thread-safe DBs the original _get_thread_db() then called
        deepcopy(self.db) — but the live psycopg C-extension Connection is not
        deep-copyable, causing TypeError.

        Fixed code returns self.db directly (no deepcopy), so this test must pass
        without raising.
        """
        db = make_db("test_get_thread_db")
        runner = ConcurrentInsertRunner(db=db, dataset=MagicMock(), normalize=False)

        with db.init():
            assert db.conn is not None
            result = runner._get_thread_db()  # TypeError here on original code

        assert result is db


# ── ConcurrentInsertRunner tests ──────────────────────────────────────────────


class TestPgVectorConcurrentInsert:
    """Tests for ConcurrentInsertRunner with PgVector (reproduces issue #756)."""

    @pytest.mark.integration
    def test_concurrent_insert_e2e(self):
        """E2E regression test for issue #756 using the OpenAI 50K dataset.

        Exercises the full pipeline:
          ProcessPoolExecutor(spawn) → pickle runner → subprocess task()
          → with self.db.init() → worker _get_thread_db() → insert batches

        FAILS on original code (TypeError: deepcopy of live psycopg connection).
        PASSES on fixed code.
        """
        dataset = Dataset.OPENAI.manager(50_000)
        dataset.prepare(DatasetSource.AliyunOSS)

        cfg = dict(DB_CONFIG)
        cfg["table_name"] = "test_e2e_insert"
        db = DB.PgVector.init_cls(
            dim=dataset.data.dim,
            db_config=cfg,
            db_case_config=PgVectorHNSWConfig(
                metric_type="COSINE",
                m=16,
                ef_construction=64,
                ef_search=64,
            ),
            drop_old=True,
        )

        runner = ConcurrentInsertRunner(db=db, dataset=dataset, normalize=True, filters=non_filter)
        count = runner.run()

        assert count == 50_000, f"Expected 50000 rows, got {count}"
        log.info(f"E2E insert completed: {count} rows")
