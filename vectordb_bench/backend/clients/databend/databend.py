"""Wrapper around the Databend vector database over VectorDB"""

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from databend_driver import BlockingDatabendClient, BlockingDatabendConnection

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import DatabendConfigDict, DatabendIndexConfig

log = logging.getLogger(__name__)


def write_arrow_stream(
    sink: Any,
    metadata: list[int],
    embeddings: list[list[float]] | np.ndarray,
    dim: int,
    id_field: str = "id",
    vector_field: str = "embedding",
) -> None:
    """Writes the rows to `sink` as an Arrow IPC stream: the ids as UInt32 and the
    embeddings as float32 lists of `dim` values, which Databend loads into `Vector(dim)`."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.shape != (len(metadata), dim):
        msg = f"expected {len(metadata)} embeddings of {dim} dims, got shape {vectors.shape}"
        raise ValueError(msg)
    table = pa.table(
        {
            id_field: pa.array(metadata, type=pa.uint32()),
            vector_field: pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), dim),
        }
    )
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)


def search_settings(search_param: dict) -> str:
    """The `SETTINGS (...)` prefix of a search query, or an empty string."""
    settings = []
    if search_param.get("refine_factor"):
        settings.append(f"vector_refine_factor = {search_param['refine_factor']}")
    if search_param.get("ef_search"):
        settings.append(f"vector_search_ef = {search_param['ef_search']}")
    return f"SETTINGS ({', '.join(settings)}) " if settings else ""


class Databend(VectorDB):
    """Databend over `databend_driver`: a table with an HNSW vector index, loaded as
    Arrow streams, searched with `ORDER BY <distance>(...) LIMIT k`."""

    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.NumGE]

    def __init__(
        self,
        dim: int,
        db_config: DatabendConfigDict,
        db_case_config: DatabendIndexConfig,
        collection_name: str = "DatabendVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.database_name = db_config["database"]
        self.table_name = collection_name
        self.dim = dim

        self.index_param = self.case_config.index_param()
        self._search_settings = search_settings(self.case_config.search_param())
        self._distance_func = "l2_distance" if self.index_param["metric_type"] == "l2" else "cosine_distance"
        self._where_clause = ""

        self._index_name = "databend_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        if drop_old:
            self.conn = self._create_connection()
            log.info(f"Databend client drop table : {self.table_name}")
            self._drop_table()
            self._create_table()
            if self.case_config.create_index_before_load:
                self._create_index()
            self.conn.close()
        self.conn = None

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.conn = self._create_connection()
        try:
            yield
        finally:
            self.conn.close()
            self.conn = None

    def _create_connection(self) -> BlockingDatabendConnection:
        c = self.db_config
        sslmode = "enable" if c.get("secure") else "disable"
        dsn = f"databend://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}?sslmode={sslmode}"
        return BlockingDatabendClient(dsn).get_conn()

    def _exec(self, sql: str, action: str) -> None:
        assert self.conn is not None, "Connection is not initialized"
        try:
            self.conn.exec(sql)
        except Exception as e:
            log.warning(f"Failed to {action} on Databend table {self.database_name}.{self.table_name}: {e}")
            raise e from None

    def _drop_table(self):
        self._exec(f"DROP TABLE IF EXISTS {self.database_name}.{self.table_name}", "drop table")

    def _create_table(self):
        self._exec(
            f"CREATE TABLE IF NOT EXISTS {self.database_name}.{self.table_name} "
            f"({self._primary_field} UInt32, {self._vector_field} Vector({self.dim})) ENGINE = Fuse",
            "create table",
        )

    def _create_index(self):
        refine = " refine = true" if self.index_param["refine_factor"] else ""
        self._exec(
            f"CREATE VECTOR INDEX IF NOT EXISTS {self._index_name} "
            f"ON {self.database_name}.{self.table_name} ({self._vector_field}) "
            f"m = {self.index_param['m']} ef_construct = {self.index_param['ef_construct']} "
            f"distance = {self.index_param['metric_type']}{refine}",
            "create vector index",
        )

    def optimize(self, data_size: int | None = None):
        """Builds what the load left to do: the index when it is created after the
        load, the k-means layout when configured, and compacts the table."""
        table = f"{self.database_name}.{self.table_name}"
        if self.case_config.create_index_after_load:
            self._create_index()
        if self.case_config.recluster:
            self._exec(f"ALTER TABLE {table} CLUSTER BY ({self._vector_field})", "set cluster key")
            self._exec(f"ALTER TABLE {table} RECLUSTER FINAL", "recluster")
        else:
            self._exec(f"OPTIMIZE TABLE {table} ALL", "optimize")

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        assert self.conn is not None, "Connection is not initialized"

        if not metadata:
            return 0, None
        # Binary columns, as other clients load: formatting every float as text and
        # parsing it again on the server dominated the load time.
        fd, path = tempfile.mkstemp(prefix="vdbbench_databend_", suffix=".arrows")
        try:
            with pa.OSFile(path, "wb") as sink:
                write_arrow_stream(sink, metadata, embeddings, self.dim, self._primary_field, self._vector_field)
            self.conn.load_file(
                f"INSERT INTO {self.database_name}.{self.table_name} "
                "FROM @_databend_load FILE_FORMAT = (type = arrow_stream)",
                path,
                # Sent as the body of one streaming load request, without a stage upload.
                "streaming",
            )
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Databend table ({self.table_name}), error: {e}")
            return 0, e
        finally:
            os.close(fd)
            Path(path).unlink()

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self._where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self._where_clause = f"WHERE {self._primary_field} >= {filters.int_value} "
        else:
            msg = f"Not support Filter for Databend - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        result = self.conn.query_all(
            f"{self._search_settings}SELECT {self._primary_field} "
            f"FROM {self.database_name}.{self.table_name} "
            f"{self._where_clause}"
            f"ORDER BY {self._distance_func}({self._vector_field}, {query}::Vector({self.dim})) "
            f"LIMIT {k}",
        )
        return [int(row.values()[0]) for row in result]
