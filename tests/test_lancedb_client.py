"""Unit tests for LanceDB client insert semantics.

Focus: ConcurrentInsertRunner + thread_safe=True must not duplicate IDs when
a multi-fragment insert commits a prefix and then fails.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")

from vectordb_bench.backend.clients.api import PartialInsertError  # noqa: E402
from vectordb_bench.backend.clients.lancedb import lancedb as lancedb_module  # noqa: E402
from vectordb_bench.backend.clients.lancedb.config import LanceDBNoIndexConfig  # noqa: E402
from vectordb_bench.backend.clients.lancedb.lancedb import LanceDB  # noqa: E402
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner  # noqa: E402


def _make_client(tmp_path, *, batch_size: int, monkeypatch) -> LanceDB:
    monkeypatch.setattr(lancedb_module, "LANCEDB_BATCH_SIZE", batch_size)
    return LanceDB(
        dim=2,
        db_config={"uri": str(tmp_path / "db")},
        db_case_config=LanceDBNoIndexConfig(),
        collection_name="bench",
        drop_old=True,
    )


def test_insert_embeddings_partial_commit_is_non_retryable(tmp_path, monkeypatch):
    """If the first fragment commits and a later add fails, report the
    committed prefix as PartialInsertError (non-retryable) instead of (0, err).
    """
    db = _make_client(tmp_path, batch_size=2, monkeypatch=monkeypatch)
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    metadata = [1, 2, 3, 4]

    with db.init():
        real_add = db.table.add
        calls = {"n": 0}

        def flaky_add(batch):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_add(batch)
            raise RuntimeError("simulated fragment commit failure")

        db.table.add = flaky_add
        inserted, err = db.insert_embeddings(embeddings, metadata)

    assert inserted == 2
    assert isinstance(err, PartialInsertError)
    assert err.non_retryable is True
    assert err.inserted_count == 2

    with db.init():
        assert db.table.count_rows() == 2


def test_insert_embeddings_full_success_returns_all_rows(tmp_path, monkeypatch):
    db = _make_client(tmp_path, batch_size=2, monkeypatch=monkeypatch)
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    metadata = [10, 11, 12]

    with db.init():
        inserted, err = db.insert_embeddings(embeddings, metadata)
        assert err is None
        assert inserted == 3
        assert db.table.count_rows() == 3


def test_concurrent_insert_exact_row_count_with_thread_safe(tmp_path, monkeypatch):
    """Regression: ConcurrentInsertRunner + thread_safe must insert each ID once.

    Uses NUM_PER_BATCH > LANCEDB_BATCH_SIZE so inserts span multiple fragments.
    """
    monkeypatch.setattr(lancedb_module, "LANCEDB_BATCH_SIZE", 2)
    db = LanceDB(
        dim=2,
        db_config={"uri": str(tmp_path / "db")},
        db_case_config=LanceDBNoIndexConfig(),
        collection_name="bench",
        drop_old=True,
    )
    assert db.thread_safe is True

    class Data:
        train_id_field = "id"
        train_vector_field = "vector"
        scalar_labels_file_separated = False

    class Dataset:
        data = Data()

        def iter_batches(self, batch_size: int):
            import pandas as pd

            # Two runner batches of 3 rows each; each batch spans 2 Lance fragments.
            rows = [
                {"id": i, "vector": [float(i), float(i) + 0.1]}
                for i in range(6)
            ]
            for start in range(0, len(rows), batch_size):
                yield pd.DataFrame(rows[start : start + batch_size])

    runner = ConcurrentInsertRunner(
        db,
        Dataset(),
        normalize=False,
        max_workers=2,
        batch_size=3,
    )
    count = runner.task()
    assert count == 6

    with db.init():
        assert db.table.count_rows() == 6
        ids = sorted(db.table.to_pandas()["id"].tolist())
        assert ids == [0, 1, 2, 3, 4, 5]


def test_concurrent_runner_does_not_retry_lancedb_partial_insert(monkeypatch):
    """PartialInsertError from LanceDB must stop the batch retry loop."""
    from vectordb_bench.backend.runner import concurrent_runner as concurrent_runner_module

    class FakeDB:
        name = "LanceDB"
        thread_safe = True

        def __init__(self):
            self.calls = 0

        @contextmanager
        def init(self):
            yield

        def insert_embeddings(self, embeddings, metadata, labels_data=None):
            self.calls += 1
            # Simulate first fragment (2 rows) committed, second failed.
            return 2, PartialInsertError(
                "partial fragment commit",
                inserted_count=2,
                cause=RuntimeError("add failed"),
            )

    monkeypatch.setattr(concurrent_runner_module.time, "sleep", lambda _seconds: None)

    runner = ConcurrentInsertRunner.__new__(ConcurrentInsertRunner)
    db = FakeDB()
    with pytest.raises(RuntimeError, match="Non-retryable insert failure"):
        runner._insert_batch_with_retry(
            db,
            embeddings=[[0.1, 0.2]] * 4,
            metadata=[1, 2, 3, 4],
        )
    assert db.calls == 1
