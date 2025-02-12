import os
from typing import Annotated, TypedDict, Unpack
import click
from pydantic import SecretStr
from ..api import IndexType, MetricType

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from vectordb_bench.backend.clients import DB

class OceanBaseTypedDict(CommonTypedDict):
    host: Annotated[
        str, click.option("--host", type=str, help="OceanBase host", default="")
    ]
    unixsock: Annotated[
        str, click.option("--unixsock", type=str, help="Unix socket file path", default="")
    ]
    user: Annotated[
        str, click.option("--user", type=str, help="OceanBase username", required=True)
    ]
    password: Annotated[
        str,
        click.option("--password",
                     type=str,
                     help="OceanBase database password",
                     default=lambda: os.environ.get("OB_PASSWORD", ""),
                     ),
    ]
    database: Annotated[
        str, click.option("--database", type=str, help="DataBase name", required=True)
    ]
    port: Annotated[
        int, click.option("--port", type=int, help="OceanBase port", required=True)
    ]

class OceanBaseHNSWTypedDict(CommonTypedDict, OceanBaseTypedDict, HNSWFlavor3):
    ...

@cli.command()
@click_parameter_decorators_from_typed_dict(OceanBaseHNSWTypedDict)
def OceanBaseHNSW(**parameters: Unpack[OceanBaseHNSWTypedDict]):
    from .config import OceanBaseConfig, OceanBaseHNSWConfig

    run(
        db=DB.OceanBase,
        db_config=OceanBaseConfig(
            db_label=parameters["db_label"],
            user=SecretStr(parameters["user"]),
            password=SecretStr(parameters["password"]),
            unix_socket=parameters["unixsock"],
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
        ),
        db_case_config=OceanBaseHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )