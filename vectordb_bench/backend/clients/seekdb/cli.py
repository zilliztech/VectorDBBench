import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class SeekDBTypedDict(CommonTypedDict):
    host: Annotated[str, click.option("--host", type=str, help="SeekDB host", required=True)]
    user: Annotated[str, click.option("--user", type=str, help="SeekDB username", required=True)]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="SeekDB password",
            default=lambda: os.environ.get("SEEKDB_PASSWORD", ""),
        ),
    ]
    database: Annotated[str, click.option("--database", type=str, help="Database name", required=True)]
    port: Annotated[int, click.option("--port", type=int, help="SeekDB port", default=3306, show_default=True)]


class SeekDBHNSWTypedDict(SeekDBTypedDict, HNSWFlavor3): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(SeekDBHNSWTypedDict)
def SeekDBHNSW(**parameters: Unpack[SeekDBHNSWTypedDict]):
    """Run VectorDBBench against SeekDB with an HNSW index."""
    from ..api import IndexType
    from .config import SeekDBConfig, SeekDBHNSWConfig

    run(
        DB.SeekDB,
        SeekDBConfig(
            db_label=parameters["db_label"],
            user=SecretStr(parameters["user"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
        ),
        SeekDBHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            index=IndexType.HNSW,
        ),
        **parameters,
    )
