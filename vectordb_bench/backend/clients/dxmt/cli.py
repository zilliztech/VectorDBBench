from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

from .config import SEARCH_MODES

DBTYPE = DB.DXMT


class DXMTTypedDict(CommonTypedDict):
    url: Annotated[
        str,
        click.option(
            "--url",
            type=str,
            default="http://localhost:9999",
            show_default=True,
            help="DXMT server URL",
        ),
    ]
    api_key: Annotated[
        str,
        click.option(
            "--api-key",
            type=str,
            envvar="DXMT_API_KEY",
            default="",
            show_default=False,
            help="API key, when the server sets DXMT_API_KEY  [env: DXMT_API_KEY]",
        ),
    ]
    timeout_sec: Annotated[
        int,
        click.option("--timeout-sec", type=int, default=600, show_default=True, help="Per-request timeout in seconds"),
    ]
    search_mode: Annotated[
        str | None,
        click.option(
            "--search-mode",
            type=click.Choice(SEARCH_MODES, case_sensitive=False),
            default=None,
            help="Per-query search effort; the server default is normal",
        ),
    ]


class DXMTHNSWTypedDict(DXMTTypedDict):
    m: Annotated[
        int,
        click.option("--m", type=int, default=16, show_default=True, help="HNSW m, edges per node"),
    ]
    ef_construction: Annotated[
        int,
        click.option("--ef-construction", type=int, default=200, show_default=True, help="HNSW ef_construction"),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, default=128, show_default=True, help="HNSW ef_search"),
    ]


def _db_config(parameters: dict):
    from .config import DXMTConfig

    return DXMTConfig(
        db_label=parameters["db_label"],
        note=parameters["note"],
        url=parameters["url"],
        api_key=SecretStr(parameters["api_key"]) if parameters["api_key"] else None,
        timeout_sec=parameters["timeout_sec"],
    )


def _search_mode(parameters: dict) -> str | None:
    return parameters["search_mode"].lower() if parameters["search_mode"] else None


@cli.command()
@click_parameter_decorators_from_typed_dict(DXMTTypedDict)
def DXMT(**parameters: Unpack[DXMTTypedDict]):
    """DXMT with the index and parameters the server picks for the table."""
    from .config import DXMTAutoIndexConfig

    run(
        db=DBTYPE,
        db_config=_db_config(parameters),
        db_case_config=DXMTAutoIndexConfig(search_mode=_search_mode(parameters)),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(DXMTHNSWTypedDict)
def DXMTHNSW(**parameters: Unpack[DXMTHNSWTypedDict]):
    """DXMT with explicit HNSW graph parameters."""
    from .config import DXMTHNSWConfig

    run(
        db=DBTYPE,
        db_config=_db_config(parameters),
        db_case_config=DXMTHNSWConfig(
            search_mode=_search_mode(parameters),
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
