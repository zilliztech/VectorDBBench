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
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.pgvector.config import (
    PgVectorFLATConfig,
    PgVectorHNSWConfig,
    PgVectorIVFFlatConfig,
    _pgvector_case_config,
)
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


def make_flat_config(**kwargs) -> PgVectorFLATConfig:
    return PgVectorFLATConfig(
        metric_type="COSINE",
        **kwargs,
    )


def make_db(table_name: str = "test_pgvector", drop_old: bool = True, case_config=None) -> DB.PgVector.init_cls:
    cfg = dict(DB_CONFIG)
    cfg["table_name"] = table_name
    if case_config is None:
        case_config = make_hnsw_config()
    return DB.PgVector.init_cls(
        dim=DIM,
        db_config=cfg,
        db_case_config=case_config,
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


# ── PgVectorFLAT tests ────────────────────────────────────────────────────────


class TestPgVectorFLAT:
    """Unit and mock tests for pgvectorFLAT implementation."""

    def test_flat_config_defaults(self):
        cfg = PgVectorFLATConfig(metric_type="COSINE")
        assert cfg.index == IndexType.Flat
        assert cfg.create_index_before_load is False
        assert cfg.create_index_after_load is False
        assert cfg.session_param() == {"session_options": []}

        idx_p = cfg.index_param()
        assert idx_p["index_type"] == "FLAT"
        assert idx_p["index_creation_with_options"] == []
        assert idx_p["quantization_type"] == "vector"

    def test_flat_case_config_resolution(self):
        resolved_cls = DB.PgVector.case_config_cls(IndexType.Flat)
        assert resolved_cls == PgVectorFLATConfig
        assert _pgvector_case_config.get(IndexType.Flat) == PgVectorFLATConfig

    def test_flat_init_without_index_flags(self):
        """PgVector.__init__ must not raise RuntimeError when both index flags are False for FLAT."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        with (
            MagicMock(),
            pytest.MonkeyPatch.context() as mp,
        ):
            mp.setattr(DB.PgVector.init_cls, "_create_connection", lambda self, **kw: (mock_conn, mock_cursor))
            flat_cfg = make_flat_config()

            # Should initialize cleanly without raising RuntimeError
            db = DB.PgVector.init_cls(
                dim=DIM,
                db_config=dict(DB_CONFIG),
                db_case_config=flat_cfg,
                drop_old=False,
            )
            assert db.case_config.index == IndexType.Flat

    def test_create_index_defensive_skip_on_flat(self):
        """_create_index must log and return early when index_type is FLAT."""
        mock_conn = MagicMock()
        mock_conn.pgconn._encoding = "utf-8"
        mock_cursor = MagicMock(connection=mock_conn)
        flat_cfg = make_flat_config()

        db = DB.PgVector.init_cls.__new__(DB.PgVector.init_cls)
        db.name = "PgVector"
        db.conn = mock_conn
        db.cursor = mock_cursor
        db.case_config = flat_cfg
        db._index_name = "pgvector_index"
        db.connect_config = DB_CONFIG["connect_config"]

        # Call _create_index directly
        db._create_index()

        # Execute should not have been called for CREATE INDEX
        for call in mock_cursor.execute.call_args_list:
            sql_str = str(call[0][0]) if call[0] else ""
            assert "CREATE INDEX" not in sql_str

    def test_post_insert_drops_managed_index_on_flat(self):
        """_post_insert / optimize must execute _drop_index for FLAT config to ensure managed index is removed."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock(connection=None)
        flat_cfg = make_flat_config()

        db = DB.PgVector.init_cls.__new__(DB.PgVector.init_cls)
        db.name = "PgVector"
        db.conn = mock_conn
        db.cursor = mock_cursor
        db.case_config = flat_cfg
        db._index_name = "pgvector_index"

        db.optimize()

        # Confirm _drop_index executed DROP INDEX IF EXISTS pgvector_index
        drop_executed = False
        for call in mock_cursor.execute.call_args_list:
            sql_str = str(call[0][0]) if call[0] else ""
            if "DROP INDEX" in sql_str and "pgvector_index" in sql_str:
                drop_executed = True
                break

        assert drop_executed is True

    def test_cli_command_registration(self):
        from vectordb_bench.cli.vectordbbench import cli

        command_names = [cmd for cmd in cli.commands]
        assert "pgvectorflat" in command_names

    @pytest.mark.integration
    def test_flat_e2e_no_ann_index(self):
        """Integration test with live PostgreSQL/pgvector instance.

        Verifies:
        - Insertion into pgvector table.
        - Calling optimize() under FLAT configuration removes pgvector_index if present.
        - Verification via pg_indexes that pgvector_index is not present.
        - Verification via EXPLAIN that pgvector_index is not used in the plan.
        - Search query succeeds and returns valid IDs.
        """
        cfg = dict(DB_CONFIG)
        cfg["connect_config"]["port"] = 5433  # use local test container if running
        cfg["table_name"] = "test_flat_e2e"

        # 1. First build an HNSW index on the table to test pre-existing index removal
        hnsw_db = DB.PgVector.init_cls(
            dim=DIM,
            db_config=cfg,
            db_case_config=make_hnsw_config(),
            drop_old=True,
        )
        embeddings = random_embeddings(n=100)
        metadata = list(range(100))

        with hnsw_db.init():
            hnsw_db.insert_embeddings(embeddings=embeddings, metadata=metadata)
            hnsw_db.optimize()  # creates HNSW pgvector_index

        # 2. Now initialize FLAT on the same table with drop_old=False
        flat_db = DB.PgVector.init_cls(
            dim=DIM,
            db_config=cfg,
            db_case_config=make_flat_config(),
            drop_old=False,
        )

        with flat_db.init():
            flat_db.optimize()  # must call _drop_index and skip _create_index

            # Check pg_indexes table to verify pgvector_index is absent
            flat_db.cursor.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'test_flat_e2e' AND indexname = 'pgvector_index';"
            )
            rows = flat_db.cursor.fetchall()
            assert len(rows) == 0, f"Expected pgvector_index to be dropped, found: {rows}"

            # Check EXPLAIN output to verify pgvector_index is not used
            flat_db.prepare_filter(Filter(type=FilterOp.NonFilter))
            explain_query = flat_db.cursor.execute(
                f"EXPLAIN {flat_db._search.as_string(flat_db.cursor)}",
                (embeddings[0], 10),
            ).fetchall()
            explain_text = " ".join([r[0] for r in explain_query])
            assert "pgvector_index" not in explain_text

            # Execute search
            results = flat_db.search_embedding(query=embeddings[0], k=10)
            assert len(results) == 10


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

