import importlib.resources
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np
import sqlite_vector

from ..api import MetricType, VectorDB
from .config import SQLiteVectorIndexConfig

_DISTANCE_BY_METRIC = {
    MetricType.COSINE: "COSINE",
    MetricType.L2: "L2",
    MetricType.IP: "DOT",
    MetricType.DP: "DOT",
}

_SEARCH_SQL = """
    SELECT rowid
    FROM vector_full_scan('vectors', 'embedding', ?, ?)
    """


class SQLiteVector(VectorDB):
    name = "SQLiteVector"
    thread_safe = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: SQLiteVectorIndexConfig | None,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        del collection_name, kwargs
        self.dim = dim
        self.db_path = Path(db_config["db_path"]).expanduser()
        self.connection: sqlite3.Connection | None = None
        self._operation_lock = threading.Lock()
        metric_type = db_case_config.metric_type if db_case_config is not None else MetricType.COSINE
        if metric_type is None:
            metric_type = MetricType.COSINE
        try:
            distance = _DISTANCE_BY_METRIC[metric_type]
        except KeyError:
            msg = f"Unsupported metric type: {metric_type}"
            raise ValueError(msg) from None
        self.vector_options = f"type=FLOAT32,dimension={dim},distance={distance}"
        self.extension_path = str(importlib.resources.files(sqlite_vector) / "binaries" / "vector")

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
                raise RuntimeError("SQLite-vector connection is already open")
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
        rows = [(int(row_id), vector.tobytes(), self.dim) for row_id, vector in zip(metadata, vectors, strict=True)]
        with self._operation_lock:
            connection = self._connection()
            try:
                connection.executemany(
                    "INSERT INTO vectors(id, embedding) VALUES (?, vector_as_f32(?, ?))",
                    rows,
                )
                connection.commit()
                return len(metadata), None
            except Exception as error:
                with suppress(sqlite3.Error):
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
            rows = self._connection().execute(_SEARCH_SQL, (vector.tobytes(), k))
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

    def _connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("Call init() before using the SQLite-vector client")
        return self.connection

    def _create_table(self) -> None:
        connection = self._open_connection()
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """)
            self._initialize_vector(connection)
            connection.commit()
        finally:
            connection.close()

    def _remove_database(self) -> None:
        for suffix in ("", "-journal", "-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            connection.enable_load_extension(True)
            try:
                connection.load_extension(self.extension_path)
            finally:
                connection.enable_load_extension(False)
        except Exception:
            connection.close()
            raise
        return connection

    def _connect(self) -> sqlite3.Connection:
        connection = self._open_connection()
        try:
            self._initialize_vector(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def _initialize_vector(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "SELECT vector_init('vectors', 'embedding', ?)",
            (self.vector_options,),
        ).fetchone()
