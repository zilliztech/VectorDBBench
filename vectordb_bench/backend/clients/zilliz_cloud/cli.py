from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class ZillizTypedDict(CommonTypedDict):
    uri: Annotated[
        str, click.option("--uri", type=str, help="Zilliz uri", required=True)
    ]
    user: Annotated[
        str,
        click.option("--user", type=str, help="Zilliz user", required=True),
    ]
    password: Annotated[
        str,
        click.option("--password", type=str, help="Zilliz password", required=True),
    ]
    index_type: Annotated[
        str,
        click.option("--index_type", type=str, help="Zilliz index type", required=True),
    ]
    metric_type: Annotated[
        str,
        click.option("--password", type=str, help="Zilliz password", required=True),
    ]
    level: Annotated[
        str,
        click.option("--level", type=str, help="Zilliz index level", required=True),
    ]



@cli.command()
@click_parameter_decorators_from_typed_dict(ZillizTypedDict)
def Zilliz(**parameters: Unpack[ZillizTypedDict]):
    from .config import ZillizCloudConfig, AutoIndexConfig

    run(
        db=DB.ZillizCloud,
        db_config=ZillizCloudConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=SecretStr(parameters["user"]),
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=AutoIndexConfig(
            params={parameters["limit"]},
        ),
    )
