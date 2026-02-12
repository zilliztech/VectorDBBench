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
from ..api import EmptyDBCaseConfig


class PineconeTypedDict(TypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="Pinecone API key", required=True),
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", type=str, help="Pinecone index name", required=True),
    ]
    version: Annotated[
        str,
        click.option("--version", type=str, help="Database version", default="", show_default=True),
    ]
    note: Annotated[
        str,
        click.option("--note", type=str, help="Additional notes", default="", show_default=True),
    ]


class PineconeIndexTypedDict(CommonTypedDict, PineconeTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PineconeIndexTypedDict)
def Pinecone(**parameters: Unpack[PineconeIndexTypedDict]):
    from .config import PineconeConfig

    run(
        db=DB.Pinecone,
        db_config=PineconeConfig(
            db_label=parameters["db_label"],
            version=parameters["version"],
            note=parameters["note"],
            api_key=SecretStr(parameters["api_key"]),
            index_name=parameters["index_name"],
        ),
        db_case_config=EmptyDBCaseConfig(),
        **parameters,
    )
