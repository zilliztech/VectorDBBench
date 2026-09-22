"""Regression tests for issue #831 — reloading results fails validation.

After a pgvector run that used an empty password (local trust/peer auth), the
saved result JSON carries ``password: ""``. When the results page rehydrates the
config via ``db.config_cls(**task_config["db_config"])`` (models.TestResult.
read_file), ``DBConfig.not_empty_field`` rejected the empty password with
``Value error, Empty field(s): password``. The same shape breaks the sibling
postgres-connection-string configs (pgvectorscale, alloydb).

These tests do not require a live database — they exercise the exact
reconstruction step read_file performs on the saved db_config dict.

Usage:
  pytest tests/test_pgvector_config_roundtrip.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.alloydb.config import AlloyDBConfig
from vectordb_bench.backend.clients.pgvector.config import PgVectorConfig
from vectordb_bench.backend.clients.pgvectorscale.config import PgVectorScaleConfig

if TYPE_CHECKING:
    from vectordb_bench.backend.clients.api import DBConfig

# The postgres-connection-string family that shares the DBConfig empty-field guard.
PG_FAMILY = [
    (DB.PgVector, PgVectorConfig),
    (DB.PgVectorScale, PgVectorScaleConfig),
    (DB.AlloyDB, AlloyDBConfig),
]
PG_FAMILY_IDS = ["pgvector", "pgvectorscale", "alloydb"]


@pytest.mark.parametrize(("db", "config_cls"), PG_FAMILY, ids=PG_FAMILY_IDS)
def test_reload_with_empty_password_present(db: DB, config_cls: type[DBConfig]):
    """A saved config with password="" must rehydrate (issue #831)."""
    saved = {"db_label": "", "password": "", "version": "", "note": ""}
    cfg = db.config_cls(**saved)
    assert isinstance(cfg, config_cls)
    assert cfg.password.get_secret_value() == ""


@pytest.mark.parametrize(("db", "config_cls"), PG_FAMILY, ids=PG_FAMILY_IDS)
def test_reload_with_password_absent(db: DB, config_cls: type[DBConfig]):
    """Older result files omit subclass fields entirely — still rehydrate."""
    saved = {"db_label": "", "version": "", "note": ""}
    cfg = db.config_cls(**saved)
    assert isinstance(cfg, config_cls)
    assert cfg.password.get_secret_value() == ""
    assert cfg.db_name  # non-empty default so downstream connection strings hold


def test_non_credential_empty_field_still_rejected():
    """Negative control: the empty-field guard must still fire for other fields."""
    with pytest.raises(ValueError, match=r"Empty field.*host"):
        PgVectorConfig(db_label="", password="x", version="", note="", host="")  # noqa: S106
