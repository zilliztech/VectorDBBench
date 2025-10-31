import unittest
from unittest.mock import patch

from click.testing import CliRunner
from pydantic import SecretStr

from vectordb_bench.backend.clients.databend import cli as databend_cli


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
