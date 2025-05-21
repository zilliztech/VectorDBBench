from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run


class TiDBTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            default="root",
            show_default=True,
            required=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            default="",
            show_default=True,
            help="Password",
        ),
    ]
    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            default="127.0.0.1",
            show_default=True,
            required=True,
            help="Db host",
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=4000,
            show_default=True,
            required=True,
            help="Db Port",
        ),
    ]
    db_name: Annotated[
        str,
        click.option(
            "--db-name",
            type=str,
            default="test",
            show_default=True,
            required=True,
            help="Db name",
        ),
    ]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            default=False,
            show_default=True,
            is_flag=True,
            help="Enable or disable SSL, for TiDB Serverless SSL must be enabled",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TiDBTypedDict)
def TiDB(
    **parameters: Unpack[TiDBTypedDict],
):
    from .config import TiDBConfig, TiDBIndexConfig

    run(
        db=DB.TiDB,
        db_config=TiDBConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            ssl=parameters["ssl"],
        ),
        db_case_config=TiDBIndexConfig(),
        **parameters,
    )
