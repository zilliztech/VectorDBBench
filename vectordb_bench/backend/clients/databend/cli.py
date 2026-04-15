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
from .config import DatabendIndexConfig


class DatabendTypedDict(TypedDict):
    password: Annotated[str, click.option("--password", type=str, help="DB password")]
    host: Annotated[str, click.option("--host", type=str, help="DB host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=8000, help="DB Port")]
    user: Annotated[int, click.option("--user", type=str, default="root", help="DB user")]


class DatabendHNSWTypedDict(CommonTypedDict, DatabendTypedDict, HNSWFlavor2): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(DatabendHNSWTypedDict)
def Databend(**parameters: Unpack[DatabendHNSWTypedDict]):
    from .config import DatabendConfig

    run(
        db=DB.Databend,
        db_config=DatabendConfig(
            db_label=parameters["db_label"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=DatabendIndexConfig(
            metric_type=None,
            m=parameters["m"],
            ef_construct=parameters["ef_construction"],
        ),
        **parameters,
    )
