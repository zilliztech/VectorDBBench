"""Every DB must resolve config_cls/init_cls or raise ModuleNotFoundError.
Anything else breaks the Run Test page's missing-optional-dep hint path.
"""

import pytest

from vectordb_bench.backend.clients import DB


@pytest.mark.parametrize("db", list(DB), ids=lambda d: d.name)
def test_db_resolves_or_missing_module(db):
    for attr in ("config_cls", "init_cls"):
        try:
            getattr(db, attr)
        except ModuleNotFoundError as e:
            assert e.name, f"{db.name}.{attr}: ModuleNotFoundError has no .name"
            return
