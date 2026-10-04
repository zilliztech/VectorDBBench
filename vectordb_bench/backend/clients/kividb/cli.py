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
from .config import KiviDBHNSWConfig


class KiviDBTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="KiviDB host", required=True)]
    password: Annotated[str, click.option("--password", type=str, help="KiviDB password")]
    port: Annotated[int, click.option("--port", type=int, default=6380, show_default=True, help="KiviDB port")]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            is_flag=True,
            show_default=True,
            default=False,
            help="Connect over TLS (needs a -tls or -full KiviDB build)",
        ),
    ]


class KiviDBHNSWTypedDict(CommonTypedDict, KiviDBTypedDict, HNSWFlavor2): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(KiviDBHNSWTypedDict)
def KiviDB(**parameters: Unpack[KiviDBHNSWTypedDict]):
    from .config import KiviDBConfig

    case_config = {"ef_runtime": parameters["ef_runtime"]}
    if parameters["m"] is not None:
        case_config["M"] = parameters["m"]
    if parameters["ef_construction"] is not None:
        case_config["ef_construction"] = parameters["ef_construction"]

    run(
        db=DB.KiviDB,
        db_config=KiviDBConfig(
            db_label=parameters["db_label"],
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            ssl=parameters["ssl"],
        ),
        db_case_config=KiviDBHNSWConfig(**case_config),
        **parameters,
    )
