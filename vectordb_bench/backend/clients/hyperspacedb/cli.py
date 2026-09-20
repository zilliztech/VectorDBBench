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

DBTYPE = DB.HyperspaceDB


class HyperspaceDBTypeDict(CommonTypedDict):
    host: Annotated[
        str,
        click.option("--host", type=str, help="HyperspaceDB host", default="localhost:50051"),
    ]
    api_key: Annotated[
        str | None,
        click.option("--api-key", type=str, help="HyperspaceDB API Key", required=False),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, help="HNSW Maximum Neighbors", default=16),
    ]
    ef_construction: Annotated[
        int,
        click.option("--ef-construction", type=int, help="HNSW efConstruction", default=100),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, help="HNSW efSearch", default=100),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(HyperspaceDBTypeDict)
def HyperspaceDB(**parameters: Unpack[HyperspaceDBTypeDict]):
    from .config import HyperspaceDBConfig, HyperspaceDBIndexConfig

    run(
        db=DBTYPE,
        db_config=HyperspaceDBConfig(
            host=parameters["host"],
            api_key=SecretStr(parameters["api_key"]) if parameters["api_key"] else None,
        ),
        db_case_config=HyperspaceDBIndexConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
