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


DBTYPE = DB.Chroma


class ChromaTypeDict(CommonTypedDict):
    host: Annotated[
        str,
        click.option("--host", type=str, help="Chroma host")
        ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="Chroma port", default=8000)
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, help="HNSW Maximum Neighbors", default=16)
    ]
    ef_construct: Annotated[
        int,
        click.option("--ef-construct", type=int, help="HNSW efConstruct", default=100)
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, help="HNSW efSearch", default=100)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(ChromaTypeDict)
def Chroma(**parameters: Unpack[ChromaTypeDict]):
    from .config import ChromaConfig, ChromaIndexConfig
    run(
        db=DBTYPE,
        db_config=ChromaConfig(host=SecretStr(parameters["host"]),
                               port=parameters["port"]),
        db_case_config=ChromaIndexConfig(
            m=parameters["m"],
            ef_construct=parameters["ef_construct"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
