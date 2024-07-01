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


class AWSOpenSearchTypedDict(TypedDict):
    host: Annotated[
        str, click.option("--host", type=str, help="Db host", required=True)
    ]
    port: Annotated[int, click.option("--port", type=int, default=443, help="Db Port")]
    user: Annotated[str, click.option("--user", type=str, default="admin", help="Db User")]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]


class AWSOpenSearchHNSWTypedDict(CommonTypedDict, AWSOpenSearchTypedDict, HNSWFlavor2):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AWSOpenSearchHNSWTypedDict)
def AWSOpenSearch(**parameters: Unpack[AWSOpenSearchHNSWTypedDict]):
    from .config import AWSOpenSearchConfig, AWSOpenSearchIndexConfig
    run(
        db=DB.AWSOpenSearch,
        db_config=AWSOpenSearchConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=AWSOpenSearchIndexConfig(
        ),
        **parameters,
    )
