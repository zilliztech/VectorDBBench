from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor2,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class MemoryDBTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    port: Annotated[int, click.option("--port", type=int, default=6379, help="Db Port")]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            is_flag=True,
            show_default=True,
            default=True,
            help="Enable or disable SSL for MemoryDB",
        ),
    ]
    ssl_ca_certs: Annotated[
        str,
        click.option(
            "--ssl-ca-certs",
            show_default=True,
            help="Path to certificate authority file to use for SSL",
        ),
    ]
    cmd: Annotated[
        bool,
        click.option(
            "--cmd",
            is_flag=True,
            show_default=True,
            default=False,
            help=(
                "Cluster Mode Disabled (CMD), use this flag when testing locally on a single node instance."
                " In production, MemoryDB only supports cluster mode (CME)"
            ),
        ),
    ]
    insert_batch_size: Annotated[
        int,
        click.option(
            "--insert-batch-size",
            type=int,
            default=10,
            help="Batch size for inserting data. Adjust this as needed, but don't make it too big",
        ),
    ]


class MemoryDBHNSWTypedDict(CommonTypedDict, MemoryDBTypedDict, HNSWFlavor2): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MemoryDBHNSWTypedDict)
def MemoryDB(**parameters: Unpack[MemoryDBHNSWTypedDict]):
    from .config import MemoryDBConfig, MemoryDBHNSWConfig

    run(
        db=DB.MemoryDB,
        db_config=MemoryDBConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
            ssl=parameters["ssl"],
            ssl_ca_certs=parameters["ssl_ca_certs"],
            cmd=parameters["cmd"],
        ),
        db_case_config=MemoryDBHNSWConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_runtime=parameters["ef_runtime"],
            insert_batch_size=parameters["insert_batch_size"],
        ),
        **parameters,
    )
