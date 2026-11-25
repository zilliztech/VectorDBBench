import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.databend import cli as databend_cli
from vectordb_bench.backend.clients.databend.config import DatabendIndexConfig
from vectordb_bench.backend.clients.databend.databend import Databend
from vectordb_bench.backend.clients.hologres.config import HologresIndexConfig


def run_cli(args: list[str]) -> dict:
    """The keyword arguments the CLI passes to `run`."""
    with patch.object(databend_cli, "run") as run:
        result = CliRunner().invoke(databend_cli.Databend, ["--host", "localhost", *args])
    assert result.exit_code == 0, repr(result.exception)
    run.assert_called_once()
    return run.call_args.kwargs


class TestDatabendCLIPassword(unittest.TestCase):
    def assert_password(self, args: list[str], expected: str) -> None:
        with patch.object(databend_cli, "run") as run:
            result = CliRunner().invoke(
                databend_cli.Databend,
                ["--host", "localhost", "--m", "16", "--ef-construction", "200", *args],
            )

        self.assertEqual(result.exit_code, 0, repr(result.exception))
        run.assert_called_once()
        config = run.call_args.kwargs["db_config"]
        self.assertIsInstance(config.password, SecretStr)
        self.assertEqual(config.password.get_secret_value(), expected)
        self.assertEqual(config.to_dict()["password"], expected)

    def test_omitted_password_uses_empty_string(self) -> None:
        self.assert_password([], "")

    def test_explicit_empty_password_is_preserved(self) -> None:
        self.assert_password(["--password", ""], "")

    def test_non_empty_password_is_preserved(self) -> None:
        self.assert_password(["--password", "test-secret"], "test-secret")


class TestDatabendCLIToSQL(unittest.TestCase):
    def client(self, args: list[str]) -> tuple[Databend, MagicMock]:
        kwargs = run_cli(args)
        case_config = kwargs["db_case_config"]
        # Set from the dataset before the client is created.
        case_config.metric_type = MetricType.COSINE
        db = Databend(dim=2, db_config=kwargs["db_config"].to_dict(), db_case_config=case_config, collection_name="t")
        conn = MagicMock()
        conn.query_all.return_value = []
        db.conn = conn
        return db, conn

    def test_omitted_hnsw_options_use_server_defaults(self) -> None:
        db, conn = self.client([])
        db._create_index()
        self.assertEqual(
            conn.exec.call_args[0][0],
            "CREATE VECTOR INDEX IF NOT EXISTS databend_index ON default.t (embedding) distance = cosine",
        )

    def test_hnsw_options(self) -> None:
        db, conn = self.client(["--m", "16", "--ef-construction", "200", "--refine-factor", "2"])
        db._create_index()
        self.assertEqual(
            conn.exec.call_args[0][0],
            "CREATE VECTOR INDEX IF NOT EXISTS databend_index ON default.t (embedding) "
            "m = 16 ef_construct = 200 distance = cosine refine = true",
        )

    def test_zero_refine_factor_is_sent(self) -> None:
        db, conn = self.client(["--refine-factor", "0"])
        db.search_embedding([0.5, 1.5], k=10)
        self.assertTrue(conn.query_all.call_args[0][0].startswith("SETTINGS (vector_refine_factor = 0) SELECT"))

    def test_omitted_refine_factor_sends_no_setting(self) -> None:
        db, conn = self.client([])
        db.search_embedding([0.5, 1.5], k=10)
        self.assertTrue(conn.query_all.call_args[0][0].startswith("SELECT"))


class TestCaseConfigRegistry(unittest.TestCase):
    def test_hologres_and_databend(self) -> None:
        self.assertIs(DB.Hologres.case_config_cls(IndexType.Hologres_Graph), HologresIndexConfig)
        self.assertIs(DB.Databend.case_config_cls(), DatabendIndexConfig)
