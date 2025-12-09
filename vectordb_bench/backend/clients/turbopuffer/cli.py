from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class TurboPufferTypedDict(TypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="TurboPuffer API key", required=True),
    ]
    api_base_url: Annotated[
        str,
        click.option(
            "--api-base-url",
            type=str,
            help="TurboPuffer API base URL",
            required=False,
            default="https://api.turbopuffer.com",
            show_default=True,
        ),
    ]
    namespace: Annotated[
        str,
        click.option(
            "--namespace",
            type=str,
            help="TurboPuffer namespace",
            required=False,
            default="vdbbench_test",
            show_default=True,
        ),
    ]


class TurboPufferIndexTypedDict(CommonTypedDict, TurboPufferTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(TurboPufferIndexTypedDict)
def TurboPuffer(**parameters: Unpack[TurboPufferIndexTypedDict]):
    from .config import TurboPufferConfig, TurboPufferIndexConfig

    run(
        db=DB.TurboPuffer,
        db_config=TurboPufferConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]),
            api_base_url=parameters["api_base_url"],
            namespace=parameters["namespace"],
        ),
        db_case_config=TurboPufferIndexConfig(),
        **parameters,
    )
