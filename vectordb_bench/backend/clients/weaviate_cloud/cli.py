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


class WeaviateTypedDict(CommonTypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="Weaviate api key", required=True),
    ]
    url: Annotated[
        str,
        click.option("--url", type=str, help="Weaviate url", required=True),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(WeaviateTypedDict)
def Weaviate(**parameters: Unpack[WeaviateTypedDict]):
    from .config import WeaviateConfig, WeaviateIndexConfig

    run(
        db=DB.WeaviateCloud,
        db_config=WeaviateConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]),
            url=SecretStr(parameters["url"]),
        ),
        db_case_config=WeaviateIndexConfig(ef=256, efConstruction=256, maxConnections=16),
        **parameters,
    )
