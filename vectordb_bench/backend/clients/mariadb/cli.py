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


class MariaDBTypedDict(CommonTypedDict):
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

    storage_engine: Annotated[
        int,
        click.option(
            "--storage-engine",
            type=click.Choice(["InnoDB", "MyISAM"]),
            help="DB storage engine",
            required=True,
        ),
    ]


class MariaDBHNSWTypedDict(MariaDBTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="M parameter in MHNSW vector indexing",
            required=False,
        ),
    ]

    ef_search: Annotated[
        int | None,
        click.option(
            "--ef-search",
            type=int,
            help="MariaDB system variable mhnsw_min_limit",
            required=False,
        ),
    ]

    max_cache_size: Annotated[
        int | None,
        click.option(
            "--max-cache-size",
            type=int,
            help="MariaDB system variable mhnsw_max_cache_size",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MariaDBHNSWTypedDict)
def MariaDBHNSW(
    **parameters: Unpack[MariaDBHNSWTypedDict],
):
    from .config import MariaDBConfig, MariaDBHNSWConfig

    run(
        db=DB.MariaDB,
        db_config=MariaDBConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=MariaDBHNSWConfig(
            M=parameters["m"],
            ef_search=parameters["ef_search"],
            storage_engine=parameters["storage_engine"],
            max_cache_size=parameters["max_cache_size"],
        ),
        **parameters,
    )
