import os
import unittest
import uuid
from unittest.mock import MagicMock, patch
from urllib.parse import unquote, urlsplit

import numpy as np
import pyarrow as pa
import pytest

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.databend.config import DatabendIndexConfig
from vectordb_bench.backend.clients.databend.databend import (
    Databend,
    connection_dsn,
    search_settings,
    write_arrow_stream,
)
from vectordb_bench.backend.filter import FilterOp, IntFilter, non_filter


def read_stream(data) -> pa.Table:
    return pa.ipc.open_stream(data).read_all()


DB_CONFIG = {"user": "u", "password": "", "host": "h", "port": 8000, "database": "db", "secure": False}


def make_client(conn, dim: int, **case_config) -> Databend:
    """A client without a server: `__init__` connects when `drop_old` is set, so it
    is not."""
    case_config = {"m": 16, "ef_construct": 64, "metric_type": MetricType.COSINE, **case_config}
    db = Databend(
        dim=dim,
        db_config=DB_CONFIG,
        db_case_config=DatabendIndexConfig(**case_config),
        collection_name="t",
    )
    db.conn = conn
    return db


class TestConfig(unittest.TestCase):
    def test_unsupported_metric_is_rejected_before_any_table_change(self) -> None:
        for metric_type in (MetricType.IP, None):
            with patch.object(Databend, "_create_connection") as connect, self.assertRaises(ValueError):
                Databend(
                    dim=2,
                    db_config=DB_CONFIG,
                    db_case_config=DatabendIndexConfig(metric_type=metric_type),
                    collection_name="t",
                    drop_old=True,
                )
            connect.assert_not_called()

    def test_password_is_percent_encoded(self) -> None:
        # URL delimiters, which the driver would otherwise parse as the port or path.
        secret = "a/b:c@d?e#f%g"
        dsn = connection_dsn({**DB_CONFIG, "password": secret, "secure": True})
        self.assertEqual(dsn, "databend://u:a%2Fb%3Ac%40d%3Fe%23f%25g@h:8000/db?sslmode=enable")
        url = urlsplit(dsn)
        self.assertEqual((url.hostname, url.port, url.path), ("h", 8000, "/db"))
        self.assertEqual(unquote(url.password), secret)


class TestSearch(unittest.TestCase):
    def test_search_settings(self) -> None:
        self.assertEqual(search_settings({}), "")
        self.assertEqual(search_settings({"refine_factor": None, "ef_search": None}), "")
        # Zero disables refinement, which the server enables by default.
        self.assertEqual(search_settings({"refine_factor": 0}), "SETTINGS (vector_refine_factor = 0) ")
        self.assertEqual(search_settings({"refine_factor": 2}), "SETTINGS (vector_refine_factor = 2) ")
        self.assertEqual(
            search_settings({"refine_factor": 2, "ef_search": 400}),
            "SETTINGS (vector_refine_factor = 2, vector_search_ef = 400) ",
        )

    def test_search_sql(self) -> None:
        conn = MagicMock()
        conn.query_all.return_value = [MagicMock(values=lambda: (7,)), MagicMock(values=lambda: (3,))]
        db = make_client(conn, 2, refine_factor=2)
        self.assertEqual(db.search_embedding([0.5, 1.5], k=10), [7, 3])
        self.assertEqual(
            conn.query_all.call_args[0][0],
            "SETTINGS (vector_refine_factor = 2) SELECT id FROM db.t "
            "ORDER BY cosine_distance(embedding, [0.5, 1.5]::Vector(2)) LIMIT 10",
        )

    def test_filter_and_l2(self) -> None:
        conn = MagicMock()
        conn.query_all.return_value = []
        db = make_client(conn, 2, metric_type=MetricType.L2)
        self.assertEqual(Databend.supported_filter_types, [FilterOp.NonFilter, FilterOp.NumGE])
        db.prepare_filter(IntFilter(filter_rate=0.01, int_value=1000))
        db.search_embedding([0.5, 1.5], k=10)
        self.assertEqual(
            conn.query_all.call_args[0][0],
            "SELECT id FROM db.t WHERE id >= 1000 ORDER BY l2_distance(embedding, [0.5, 1.5]::Vector(2)) LIMIT 10",
        )
        db.prepare_filter(non_filter)
        db.search_embedding([0.5, 1.5], k=10)
        self.assertNotIn("WHERE", conn.query_all.call_args[0][0])


class TestOptimize(unittest.TestCase):
    def sqls(self, **case_config) -> list[str]:
        conn = MagicMock()
        make_client(conn, 2, **case_config).optimize()
        return [call[0][0] for call in conn.exec.call_args_list]

    def test_default_compacts(self) -> None:
        self.assertEqual(self.sqls(), ["OPTIMIZE TABLE db.t COMPACT"])

    def test_index_after_load(self) -> None:
        sqls = self.sqls(create_index_before_load=False, create_index_after_load=True, refine_factor=2)
        self.assertEqual(
            sqls[0],
            "CREATE VECTOR INDEX IF NOT EXISTS databend_index ON db.t (embedding) "
            "m = 16 ef_construct = 64 distance = cosine refine = true",
        )
        self.assertEqual(sqls[1:], ["REFRESH VECTOR INDEX databend_index ON db.t", "OPTIMIZE TABLE db.t COMPACT"])

    def test_unset_hnsw_options_are_omitted(self) -> None:
        sqls = self.sqls(create_index_before_load=False, create_index_after_load=True, m=None, ef_construct=None)
        self.assertEqual(
            sqls[0], "CREATE VECTOR INDEX IF NOT EXISTS databend_index ON db.t (embedding) distance = cosine"
        )

    def test_recluster(self) -> None:
        self.assertEqual(
            self.sqls(recluster=True),
            ["ALTER TABLE db.t CLUSTER BY (embedding)", "ALTER TABLE db.t RECLUSTER FINAL"],
        )


class TestArrowStream(unittest.TestCase):
    def test_values_are_float32_vectors(self) -> None:
        embeddings = [[0.1, -2.5, 3.0], [1e-8, 0.0, 7.25]]
        sink = pa.BufferOutputStream()
        write_arrow_stream(sink, [7, 3], embeddings, 3)
        table = read_stream(sink.getvalue())

        self.assertEqual(table.schema.names, ["id", "embedding"])
        self.assertEqual(table.schema.field("id").type, pa.uint32())
        self.assertEqual(table.schema.field("embedding").type, pa.list_(pa.float32(), 3))
        self.assertEqual(table.column("id").to_pylist(), [7, 3])
        # The same float32 bits as the values cast from the input.
        got = np.asarray(table.column("embedding").combine_chunks().flatten(), dtype=np.float32)
        want = np.asarray(embeddings, dtype=np.float32).ravel()
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

    def test_numpy_input_and_field_names(self) -> None:
        sink = pa.BufferOutputStream()
        write_arrow_stream(sink, [1], np.ones((1, 4), dtype=np.float64), 4, "pk", "vec")
        table = read_stream(sink.getvalue())
        self.assertEqual(table.schema.names, ["pk", "vec"])
        self.assertEqual(table.column("vec").to_pylist(), [[1.0, 1.0, 1.0, 1.0]])

    def test_wrong_shape_is_rejected(self) -> None:
        for embeddings in ([[1.0, 2.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]):
            with self.assertRaises(ValueError):
                write_arrow_stream(pa.BufferOutputStream(), [1], embeddings, 3)


class TestInsertEmbeddings(unittest.TestCase):
    def test_loads_an_arrow_stream_file(self) -> None:
        loaded = {}

        def load_file(sql, path, method):
            with open(path, "rb") as f:
                loaded.update(sql=sql, path=path, method=method, table=read_stream(f.read()))

        conn = MagicMock()
        conn.load_file.side_effect = load_file
        db = make_client(conn, 2)

        count, error = db.insert_embeddings(embeddings=[[0.5, 1.5], [2.5, 3.5]], metadata=[10, 11])

        self.assertEqual((count, error), (2, None))
        self.assertEqual(loaded["sql"], "INSERT INTO db.t FROM @_databend_load FILE_FORMAT = (type = arrow_stream)")
        self.assertEqual(loaded["method"], "streaming")
        self.assertEqual(loaded["table"].column("id").to_pylist(), [10, 11])
        self.assertEqual(loaded["table"].column("embedding").to_pylist(), [[0.5, 1.5], [2.5, 3.5]])
        # The temporary file is removed.
        self.assertFalse(os.path.exists(loaded["path"]))

    def test_failure_returns_the_error_and_removes_the_file(self) -> None:
        paths = []

        def load_file(sql, path, method):
            paths.append(path)
            raise RuntimeError("load failed")

        conn = MagicMock()
        conn.load_file.side_effect = load_file
        db = make_client(conn, 2)

        count, error = db.insert_embeddings(embeddings=[[0.5, 1.5]], metadata=[1])

        self.assertEqual(count, 0)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(os.path.exists(paths[0]))

    def test_empty_batch_loads_nothing(self) -> None:
        conn = MagicMock()
        self.assertEqual(make_client(conn, 2).insert_embeddings(embeddings=[], metadata=[]), (0, None))
        conn.load_file.assert_not_called()


@pytest.mark.integration
@pytest.mark.skipif("DATABEND_TEST_DSN" not in os.environ, reason="set DATABEND_TEST_DSN to a Databend server")
def test_insert_embeddings_into_databend() -> None:
    """Loads into a real table, and checks the rows against a text INSERT of the same values."""
    from databend_driver import BlockingDatabendClient

    conn = BlockingDatabendClient(os.environ["DATABEND_TEST_DSN"]).get_conn()
    table = f"vdbbench_arrow_{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((50, 16)).astype(np.float32)
    conn.exec(f"CREATE TABLE default.{table} (id UInt32, embedding Vector(16)) ENGINE = Fuse")
    conn.exec(f"CREATE TABLE default.{table}_text (id UInt32, embedding Vector(16)) ENGINE = Fuse")
    try:
        db = make_client(conn, 16)
        db.database_name, db.table_name = "default", table
        assert db.insert_embeddings(embeddings=embeddings.tolist(), metadata=list(range(50))) == (50, None)

        values = ", ".join(f"({i}, {e.tolist()})" for i, e in enumerate(embeddings))
        conn.exec(f"INSERT INTO default.{table}_text VALUES {values}")
        row = conn.query_row(
            f"SELECT count(*), sum(if(l2_distance(a.embedding, b.embedding) <> 0, 1, 0)) "
            f"FROM default.{table} a JOIN default.{table}_text b ON a.id = b.id"
        ).values()
        assert (row[0], row[1]) == (50, 0)
    finally:
        conn.exec(f"DROP TABLE IF EXISTS default.{table}")
        conn.exec(f"DROP TABLE IF EXISTS default.{table}_text")
