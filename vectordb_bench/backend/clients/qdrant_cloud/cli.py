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


class QdrantTypedDict(CommonTypedDict):
    url: Annotated[
        str,
        click.option("--url", type=str, help="URL connection string", required=True),
    ]
    api_key: Annotated[
        str | None,
        click.option("--api-key", type=str, help="API key for authentication", required=False),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(QdrantTypedDict)
def QdrantCloud(**parameters: Unpack[QdrantTypedDict]):
    from .config import QdrantConfig, QdrantIndexConfig

    config_params = {
        "db_label": parameters["db_label"],
        "url": SecretStr(parameters["url"]),
    }

    config_params["api_key"] = SecretStr(parameters["api_key"]) if parameters["api_key"] else None

    run(
        db=DB.QdrantCloud,
        db_config=QdrantConfig(**config_params),
        db_case_config=QdrantIndexConfig(),
        **parameters,
    )
