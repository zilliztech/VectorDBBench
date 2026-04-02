"""Tests for ConcurrentInsertRunner against a running Milvus instance.

Includes:
  - Correctness tests (threading & async backends)
  - Parameterized benchmark: serial vs concurrent across (batch_size, workers) matrix

NUM_PER_BATCH is set via os.environ before each run. Since runners execute
task() in a spawn subprocess that re-imports config, the env var takes effect.

Requires:
  - Milvus running at localhost:19530
  - Network access to download OpenAI 50K dataset

Usage:
  pytest tests/test_concurrent_runner.py -v -s     # correctness tests only
  python tests/test_concurrent_runner.py             # full benchmark matrix
"""

# ruff: noqa: T201

from __future__ import annotations

import logging
import os
import time

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.milvus.config import FLATConfig
from vectordb_bench.backend.dataset import Dataset, DatasetSource
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner, ExecutorBackend
from vectordb_bench.backend.runner.serial_runner import SerialInsertRunner

log = logging.getLogger("vectordb_bench")
log.setLevel(logging.INFO)

DATASET_SIZE = 50_000


# ── Shared helpers ──────────────────────────────────────────────────────


def get_milvus_db(collection_name: str):
    return DB.Milvus.init_cls(
        dim=1536,
        db_config={"uri": "http://localhost:19530", "user": "", "password": ""},
        db_case_config=FLATConfig(metric_type="COSINE"),
        collection_name=collection_name,
        drop_old=True,
    )


def prepare_dataset():
    dataset = Dataset.OPENAI.manager(DATASET_SIZE)
    dataset.prepare(DatasetSource.AliyunOSS)
    return dataset


def set_batch_size(batch_size: int) -> None:
    os.environ["NUM_PER_BATCH"] = str(batch_size)


def timed_run(runner: SerialInsertRunner | ConcurrentInsertRunner) -> tuple[int, float]:
    start = time.perf_counter()
    count = runner.run()
    return count, time.perf_counter() - start


# ── Correctness tests (pytest) ──────────────────────────────────────────


def test_concurrent_insert_threading():
    """Test concurrent insert with threading backend."""
    db = get_milvus_db("test_conc_threading")
    runner = ConcurrentInsertRunner(
        db=db,
        dataset=prepare_dataset(),
        normalize=False,
        max_workers=4,
        backend=ExecutorBackend.THREADING,
    )
    count = runner.run()
    assert count == DATASET_SIZE, f"Expected {DATASET_SIZE}, got {count}"


def test_concurrent_insert_async():
    """Test concurrent insert with async backend."""
    db = get_milvus_db("test_conc_async")
    runner = ConcurrentInsertRunner(
        db=db,
        dataset=prepare_dataset(),
        normalize=False,
        max_workers=4,
        backend=ExecutorBackend.ASYNC,
    )
    count = runner.run()
    assert count == DATASET_SIZE, f"Expected {DATASET_SIZE}, got {count}"


# ── Parameterized benchmark ────────────────────────────────────────────


def run_serial(batch_size: int) -> tuple[int, float]:
    set_batch_size(batch_size)
    runner = SerialInsertRunner(
        db=get_milvus_db(f"bench_serial_b{batch_size}"),
        dataset=prepare_dataset(),
        normalize=False,
    )
    return timed_run(runner)


def run_concurrent(batch_size: int, workers: int) -> tuple[int, float]:
    set_batch_size(batch_size)
    runner = ConcurrentInsertRunner(
        db=get_milvus_db(f"bench_conc_b{batch_size}_w{workers}"),
        dataset=prepare_dataset(),
        normalize=False,
        max_workers=workers,
        backend=ExecutorBackend.THREADING,
    )
    return timed_run(runner)


def bench_matrix():
    batch_sizes = [100, 500, 1000, 5000]
    worker_counts = [1, 2, 4, 8]

    conc_headers = [f"conc({w}w)" for w in worker_counts]
    speedup_headers = [f"speedup({w}w)" for w in worker_counts]
    print(f"\n{'Batch':>6} {'#Bat':>5} {'serial':>8}", end="")
    for h in conc_headers:
        print(f" {h:>10}", end="")
    for h in speedup_headers:
        print(f" {h:>12}", end="")
    print()
    print("-" * (22 + 10 * len(worker_counts) + 12 * len(worker_counts)))

    for bs in batch_sizes:
        n_batches = DATASET_SIZE // bs
        _, dur_s = run_serial(bs)

        conc_durs = []
        for w in worker_counts:
            _, dur_c = run_concurrent(bs, w)
            conc_durs.append(dur_c)

        print(f"{bs:>6} {n_batches:>5} {dur_s:>7.2f}s", end="")
        for dur_c in conc_durs:
            print(f" {dur_c:>9.2f}s", end="")
        for dur_c in conc_durs:
            print(f" {dur_s / dur_c:>11.2f}x", end="")
        print()

    # restore default
    set_batch_size(100)


if __name__ == "__main__":
    bench_matrix()
