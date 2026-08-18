import multiprocessing as mp
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from vectordb_bench import config
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.duckdb.config import DuckDBConfig, DuckDBIndexConfig
from vectordb_bench.backend.clients.duckdb.duckdb import DuckDB
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner


class SingleBatchDataset:
    class Fields:
        train_id_field = "id"
        train_vector_field = "emb"
        scalar_labels_file_separated = False

    data = Fields()

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        del batch_size
        yield pd.DataFrame(
            {
                "id": [40, 10, 30, 20],
                "emb": [
                    [10.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
            }
        )


class StreamingDataset:
    def __init__(self) -> None:
        self.batch_index = 0

    def __iter__(self) -> "StreamingDataset":
        return self

    def __next__(self) -> pd.DataFrame:
        if self.batch_index == 4:
            raise StopIteration
        start = self.batch_index * config.NUM_PER_BATCH
        self.batch_index += 1
        return pd.DataFrame(
            {
                "id": list(range(start, start + config.NUM_PER_BATCH)),
                "emb": [[1.0, 0.0, 0.0]] * config.NUM_PER_BATCH,
            }
        )


def make_client(
    path: Path,
    drop_old: bool = True,
    metric_type: MetricType = MetricType.COSINE,
) -> DuckDB:
    return DuckDB(
        dim=3,
        db_config=DuckDBConfig(db_path=str(path)).to_dict(),
        db_case_config=DuckDBIndexConfig(metric_type=metric_type),
        drop_old=drop_old,
    )


def search_in_process(client: DuckDB, query: list[float], k: int) -> list[int]:
    client.prepare_filter(non_filter)
    with client.init():
        return client.search_embedding(query, k=k)


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (MetricType.COSINE, [40, 30, 10]),
        (MetricType.L2, [30, 10, 20]),
        (MetricType.IP, [40, 30, 10]),
        (MetricType.DP, [40, 30, 10]),
    ],
)
def test_duckdb_exact_search_reopens_after_copy_round_trip(
    tmp_path: Path,
    metric_type: MetricType,
    expected: list[int],
) -> None:
    client = make_client(tmp_path / "vectors.duckdb", metric_type=metric_type)
    dataset = SingleBatchDataset()
    batch = next(dataset.iter_batches(4))

    with client.init():
        count, error = client.insert_embeddings(batch["emb"].tolist(), batch["id"].tolist())

    assert error is None
    assert count == 4

    client = deepcopy(client)
    with client.init():
        assert client.search_embedding([1.0, 0.0, 0.0], k=3) == expected


def test_duckdb_loads_through_concurrent_runner(tmp_path: Path) -> None:
    client = make_client(tmp_path / "runner.duckdb")
    runner = ConcurrentInsertRunner(
        db=client,
        dataset=SingleBatchDataset(),
        normalize=False,
        max_workers=4,
    )

    assert runner.max_workers == 1
    assert runner.task() == 4

    with client.init():
        assert client.search_embedding([1.0, 0.0, 0.0], k=2) == [40, 30]


def test_duckdb_rolls_back_failed_load(tmp_path: Path) -> None:
    client = make_client(tmp_path / "rollback.duckdb")

    def abort_load() -> None:
        with client.init():
            count, error = client.insert_embeddings([[1.0, 0.0, 0.0]], [1])
            assert error is None
            assert count == 1
            raise RuntimeError("abort load")

    with pytest.raises(RuntimeError, match="abort load"):
        abort_load()

    with client.init():
        assert client.search_embedding([1.0, 0.0, 0.0], k=1) == []


def test_duckdb_serializes_streaming_insert_threads(tmp_path: Path) -> None:
    client = make_client(tmp_path / "streaming.duckdb")
    runner = RatedMultiThreadingInsertRunner(
        rate=config.NUM_PER_BATCH * 4,
        db=client,
        dataset_iter=StreamingDataset(),
    )
    queue = mp.Queue()

    try:
        runner.run_with_rate(queue)
    finally:
        queue.close()
        queue.join_thread()

    expected_ids = set(range(config.NUM_PER_BATCH * 4))
    with client.init():
        assert set(client.search_embedding([1.0, 0.0, 0.0], k=len(expected_ids))) == expected_ids


def test_duckdb_supports_spawned_concurrent_searches(tmp_path: Path) -> None:
    client = make_client(tmp_path / "concurrent.duckdb")
    dataset = SingleBatchDataset()
    batch = next(dataset.iter_batches(4))
    with client.init():
        count, error = client.insert_embeddings(batch["emb"].tolist(), batch["id"].tolist())

    assert error is None
    assert count == 4

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        futures = [executor.submit(search_in_process, client, [1.0, 0.0, 0.0], 2) for _ in range(2)]

    assert [future.result() for future in futures] == [[40, 30], [40, 30]]
