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
from .config import ValkeyHNSWConfig


class ValkeyTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    port: Annotated[int, click.option("--port", type=int, default=6379, help="Db Port")]
    collection_name: Annotated[
        str,
        click.option(
            "--collection-name",
            type=str,
            default="vdbbench_valkey",
            show_default=True,
            help="Valkey search index and key prefix",
        ),
    ]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            is_flag=True,
            show_default=True,
            default=True,
            help="Enable or disable SSL for Valkey",
        ),
    ]
    insecure_tls: Annotated[
        bool,
        click.option(
            "--insecure-tls",
            is_flag=True,
            show_default=True,
            default=False,
            help="Disable TLS certificate verification",
        ),
    ]
    request_timeout_ms: Annotated[
        int,
        click.option(
            "--request-timeout-ms",
            type=int,
            default=600_000,
            show_default=True,
            help="GLIDE request timeout in milliseconds",
        ),
    ]
    connection_timeout_ms: Annotated[
        int,
        click.option(
            "--connection-timeout-ms",
            type=int,
            default=10_000,
            show_default=True,
            help="GLIDE connection timeout in milliseconds",
        ),
    ]
    cmd: Annotated[
        bool,
        click.option(
            "--cmd",
            is_flag=True,
            show_default=True,
            default=False,
            help="Cluster Mode Disabled (CMD) for Valkey doesn't use Cluster conn",
        ),
    ]


class ValkeyHNSWTypedDict(CommonTypedDict, ValkeyTypedDict, HNSWFlavor2): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(ValkeyHNSWTypedDict)
def Valkey(**parameters: Unpack[ValkeyHNSWTypedDict]):
    from .config import ValkeyConfig

    run(
        db=DB.Valkey,
        db_config=ValkeyConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
            collection_name=parameters["collection_name"],
            ssl=parameters["ssl"],
            insecure_tls=parameters["insecure_tls"],
            request_timeout_ms=parameters["request_timeout_ms"],
            connection_timeout_ms=parameters["connection_timeout_ms"],
            cmd=parameters["cmd"],
        ),
        db_case_config=ValkeyHNSWConfig(
            M=parameters["m"] if parameters["m"] is not None else 16,
            efConstruction=parameters["ef_construction"] if parameters["ef_construction"] is not None else 200,
            ef=parameters["ef_runtime"] if parameters["ef_runtime"] is not None else 10,
        ),
        **parameters,
    )
