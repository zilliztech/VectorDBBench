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

DBTYPE = DB.Telys


class TelysTypedDict(CommonTypedDict):
    host: Annotated[
        str,
        click.option("--host", type=str, help="telys serve host", default="127.0.0.1"),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="telys serve TCP port", default=9099),
    ]
    access_token: Annotated[
        str | None,
        click.option("--access-token", type=str, required=False,
                     help="Shared access token (required for TCP serving; omit for a trusted Unix socket)"),
    ]
    min_rows: Annotated[
        int,
        click.option("--min-rows", type=int, default=20000,
                     help="Per-partition IVF build threshold; smaller partitions stay exact"),
    ]
    target_recall: Annotated[
        float,
        click.option("--target-recall", type=float, default=0.98, help="Per-partition IVF target recall"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TelysTypedDict)
def Telys(**parameters: Unpack[TelysTypedDict]):
    from .config import TelysConfig, TelysIndexConfig

    run(
        db=DBTYPE,
        db_config=TelysConfig(
            host=parameters["host"],
            port=parameters["port"],
            access_token=SecretStr(parameters["access_token"]) if parameters["access_token"] else None,
        ),
        db_case_config=TelysIndexConfig(
            min_rows=parameters["min_rows"],
            target_recall=parameters["target_recall"],
        ),
        **parameters,
    )
