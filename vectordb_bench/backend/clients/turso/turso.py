import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import turso

from ..api import MetricType, VectorDB
from .config import TursoIndexConfig

_SEARCH_SQL_BY_METRIC = {
    MetricType.COSINE: """
        SELECT id
        FROM vectors
        ORDER BY vector_distance_cos(embedding, ?), id
        LIMIT ?
        """,
    MetricType.L2: """
        SELECT id
        FROM vectors
        ORDER BY vector_distance_l2(embedding, ?), id
        LIMIT ?
        """,
    MetricType.IP: """
        SELECT id
        FROM vectors
        ORDER BY vector_distance_dot(embedding, ?), id
        LIMIT ?
        """,
    MetricType.DP: """
        SELECT id
        FROM vectors
        ORDER BY vector_distance_dot(embedding, ?), id
        LIMIT ?
        """,
}


class Turso(VectorDB):
    name = "Turso"
    thread_safe = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TursoIndexConfig | None,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        del collection_name, kwargs
        self.dim = dim
        self.db_path = Path(db_config["db_path"]).expanduser()
        self.experimental_multiprocess_wal = db_config.get("experimental_multiprocess_wal", True)
        self.connection: turso.Connection | None = None
        self._operation_lock = threading.Lock()
        metric_type = db_case_config.metric_type if db_case_config is not None else MetricType.COSINE
        if metric_type is None:
            metric_type = MetricType.COSINE
        try:
            self.search_sql = _SEARCH_SQL_BY_METRIC[metric_type]
        except KeyError:
            msg = f"Unsupported metric type: {metric_type}"
            raise ValueError(msg) from None

        if self.db_path.exists() and self.db_path.is_dir():
            raise IsADirectoryError(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if drop_old:
            self._remove_database()
        self._create_table()

    @contextmanager
    def init(self) -> Iterator[None]:
        with self._operation_lock:
            if self.connection is not None:
                raise RuntimeError("Turso connection is already open")
            connection = self._connect()
            self.connection = connection
        try:
            yield
        finally:
            with self._operation_lock:
                try:
                    connection.close()
                finally:
                    self.connection = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        del labels_data, tenant_labels_data, kwargs
        vectors = np.asarray(embeddings, dtype="<f4")
        expected_shape = (len(metadata), self.dim)
        if vectors.shape != expected_shape:
            msg = f"Expected embeddings with shape {expected_shape}, got {vectors.shape}"
            return 0, ValueError(msg)
        with self._operation_lock:
            connection = self._connection()
            try:
                connection.executemany(
                    "INSERT INTO vectors(id, embedding) VALUES (?, ?)",
                    [(int(row_id), vector.tobytes()) for row_id, vector in zip(metadata, vectors, strict=True)],
                )
                connection.commit()
                return len(metadata), None
            except Exception as error:
                connection.rollback()
                return 0, error

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        del kwargs
        vector = np.asarray(query, dtype="<f4")
        expected_shape = (self.dim,)
        if vector.shape != expected_shape:
            msg = f"Expected query with shape {expected_shape}, got {vector.shape}"
            raise ValueError(msg)
        with self._operation_lock:
            rows = self._connection().execute(self.search_sql, (vector.tobytes(), k))
            return [int(row[0]) for row in rows.fetchall()]

    def optimize(self, data_size: int | None = None) -> None:
        del data_size
        with self._operation_lock:
            connection = self._connection()
            connection.execute("PRAGMA optimize").fetchall()
            connection.commit()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_operation_lock", None)
        state["connection"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._operation_lock = threading.Lock()

    def _connection(self) -> turso.Connection:
        if self.connection is None:
            raise RuntimeError("Call init() before using the Turso client")
        return self.connection

    def _create_table(self) -> None:
        connection = self._connect()
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """)
            connection.commit()
        finally:
            connection.close()

    def _remove_database(self) -> None:
        for suffix in ("", "-journal", "-shm", "-tshm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    def _connect(self) -> turso.Connection:
        experimental_features = "multiprocess_wal" if self.experimental_multiprocess_wal else None
        vfs = "experimental_win_iocp" if self.experimental_multiprocess_wal and sys.platform == "win32" else None
        return turso.connect(str(self.db_path), experimental_features=experimental_features, vfs=vfs)
