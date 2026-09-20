from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.LogosDB


class LogosDBTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option(
            "--uri",
            type=str,
            help="Path to LogosDB directory (local embedded DB)",
            required=False,
            default="/tmp/vectordbbench_logosdb",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LogosDBTypedDict)
def LogosDB(**parameters: Unpack[LogosDBTypedDict]):
    from .config import LogosDBConfig, LogosDBIndexConfig

    # LogosDB is documented as single-process; disable concurrent search
    # until a thread-safe concurrent runner is available.
    parameters["search_concurrent"] = False

    run(
        db=DBTYPE,
        db_config=LogosDBConfig(uri=parameters["uri"]),
        db_case_config=LogosDBIndexConfig(),
        **parameters,
    )
