from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class AliSQLTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            required=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Password",
            required=True,
        ),
    ]

    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="Db host",
            default="127.0.0.1",
        ),
    ]

    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=3306,
            help="Db Port",
        ),
    ]

    database: Annotated[
        str,
        click.option(
            "--database",
            type=str,
            help="Database name",
            default="vectordbbench",
        ),
    ]


class AliSQLHNSWTypedDict(AliSQLTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="M parameter in HNSW vector indexing",
            required=False,
        ),
    ]

    ef_search: Annotated[
        int | None,
        click.option(
            "--ef-search",
            type=int,
            help="AliSQL system variable vidx_hnsw_ef_search",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AliSQLHNSWTypedDict)
def AliSQLHNSW(
    **parameters: Unpack[AliSQLHNSWTypedDict],
):
    from .config import AliSQLConfig, AliSQLHNSWConfig

    run(
        db=DB.AliSQL,
        db_config=AliSQLConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
        ),
        db_case_config=AliSQLHNSWConfig(
            M=parameters["m"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
