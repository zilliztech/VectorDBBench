import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import duckdb
import numpy as np

from ...filter import Filter, FilterOp
from ..api import MetricType, VectorDB
from .config import DuckDBIndexConfig

_DISTANCE_FUNCTION_BY_METRIC = {
    MetricType.COSINE: "array_cosine_distance",
    MetricType.L2: "array_distance",
    MetricType.IP: "array_negative_inner_product",
    MetricType.DP: "array_negative_inner_product",
}


class DuckDB(VectorDB):
    name = "DuckDB"
    thread_safe = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DuckDBIndexConfig | None,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        del collection_name, kwargs
        if dim <= 0:
            msg = f"Embedding dimension must be positive, got {dim}"
            raise ValueError(msg)

        self.dim = dim
        self.db_path = Path(db_config["db_path"]).expanduser()
        self.threads = int(db_config.get("threads", 1))
        self.connection: duckdb.DuckDBPyConnection | None = None
        self._connection_read_only: bool | None = None
        self._active = False
        self._operation_lock = threading.Lock()

        metric_type = db_case_config.metric_type if db_case_config is not None else MetricType.COSINE
        if metric_type is None:
            metric_type = MetricType.COSINE
        try:
            distance_function = _DISTANCE_FUNCTION_BY_METRIC[metric_type]
        except KeyError:
            msg = f"Unsupported metric type: {metric_type}"
            raise ValueError(msg) from None

        self._insert_sql = f"INSERT INTO vectors SELECT unnest(?::BIGINT[]), unnest(?::FLOAT[{self.dim}][])"
        self._search_sql = (
            f"SELECT id FROM vectors ORDER BY {distance_function}(embedding, ?::FLOAT[{self.dim}]) LIMIT ?"
        )

        if self.db_path.exists() and self.db_path.is_dir():
            raise IsADirectoryError(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if drop_old:
            self._remove_database()
        self._create_table()

    @contextmanager
    def init(self) -> Iterator[None]:
        with self._operation_lock:
            if self._active:
                raise RuntimeError("DuckDB connection is already open")
            self._active = True
        try:
            yield
        except BaseException:
            with self._operation_lock:
                if self.connection is not None and self._connection_read_only is False:
                    with suppress(Exception):
                        self.connection.rollback()
            raise
        else:
            with self._operation_lock:
                if self.connection is not None and self._connection_read_only is False:
                    self.connection.commit()
        finally:
            with self._operation_lock:
                try:
                    if self.connection is not None:
                        self.connection.close()
                finally:
                    self.connection = None
                    self._connection_read_only = None
                    self._active = False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        del labels_data, tenant_labels_data, kwargs
        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
            ids = np.asarray(metadata, dtype=np.int64)
        except Exception as error:
            return 0, error

        expected_shape = (len(metadata), self.dim)
        if vectors.shape != expected_shape:
            msg = f"Expected embeddings with shape {expected_shape}, got {vectors.shape}"
            return 0, ValueError(msg)
        if ids.shape != (len(metadata),):
            msg = f"Expected metadata with shape {(len(metadata),)}, got {ids.shape}"
            return 0, ValueError(msg)

        try:
            with self._operation_lock:
                self._connection(read_only=False).execute(self._insert_sql, [ids, vectors])
        except Exception as error:
            return 0, error
        return len(metadata), None

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        del kwargs
        vector = np.asarray(query, dtype=np.float32)
        expected_shape = (self.dim,)
        if vector.shape != expected_shape:
            msg = f"Expected query with shape {expected_shape}, got {vector.shape}"
            raise ValueError(msg)

        with self._operation_lock:
            rows = self._connection(read_only=True).execute(self._search_sql, [vector, k]).fetchall()
        return [int(row[0]) for row in rows]

    def optimize(self, data_size: int | None = None) -> None:
        del data_size

    def prepare_filter(self, filters: Filter) -> None:
        if filters.type != FilterOp.NonFilter:
            msg = f"Unsupported filter for DuckDB: {filters}"
            raise ValueError(msg)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_operation_lock", None)
        state["connection"] = None
        state["_connection_read_only"] = None
        state["_active"] = False
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._operation_lock = threading.Lock()

    def _connection(self, read_only: bool) -> duckdb.DuckDBPyConnection:
        if not self._active:
            raise RuntimeError("Call init() before using the DuckDB client")
        if self.connection is None:
            connection = self._open_connection(read_only=read_only)
            if not read_only:
                connection.begin()
            self.connection = connection
            self._connection_read_only = read_only
        elif not read_only and self._connection_read_only:
            raise RuntimeError("Cannot write through a read-only DuckDB connection")
        return self.connection

    def _create_table(self) -> None:
        connection = self._open_connection(read_only=False)
        try:
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS vectors (id BIGINT PRIMARY KEY, embedding FLOAT[{self.dim}] NOT NULL)"
            )
        finally:
            connection.close()

    def _remove_database(self) -> None:
        self.db_path.unlink(missing_ok=True)
        Path(f"{self.db_path}.wal").unlink(missing_ok=True)

    def _open_connection(self, read_only: bool) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            str(self.db_path),
            read_only=read_only,
            config={"threads": self.threads},
        )
