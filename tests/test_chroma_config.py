from types import SimpleNamespace

import pytest
from pydantic import SecretStr

pytest.importorskip("chromadb")

from vectordb_bench.backend.clients.chroma.config import ChromaConfig
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.components.run_test.dbConfigSetting import dbConfigSettingItem


@pytest.mark.parametrize(
    ("kwargs", "expected_host", "expected_port"),
    [
        ({}, "localhost", 8000),
        ({"host": "localhost"}, "localhost", 8000),
        ({"host": "chroma.internal"}, "chroma.internal", 8000),
        ({"host": SecretStr("chroma.internal")}, "chroma.internal", 8000),
        ({"port": 9000}, "localhost", 9000),
    ],
)
def test_chroma_connection_config(kwargs, expected_host, expected_port):
    assert ChromaConfig(**kwargs).to_dict() == {"host": expected_host, "port": expected_port}


def test_chroma_host_form_default_stays_readable():
    assert ChromaConfig.model_json_schema()["properties"]["host"]["default"] == "localhost"


def test_chroma_form_submits_default_host():
    form = SimpleNamespace(
        markdown=lambda *args, **kwargs: None,
        text_input=lambda label, **kwargs: kwargs["value"],
        text_area=lambda label, **kwargs: kwargs["value"],
    )
    form.columns = lambda count: [form] * count

    values = dbConfigSettingItem(form, DB.Chroma)

    assert values["host"] == "localhost"
    assert ChromaConfig(**values).to_dict() == {"host": "localhost", "port": 8000}
