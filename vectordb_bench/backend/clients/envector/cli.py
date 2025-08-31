from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.EnVector


class EnVectorTypedDict(TypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    

class EnVectorFlatIndexTypedDict(CommonTypedDict, EnVectorTypedDict): ...


@cli.command(name="envectorflat")
@click_parameter_decorators_from_typed_dict(EnVectorFlatIndexTypedDict)
def EnVectorFlat(**parameters: Unpack[EnVectorFlatIndexTypedDict]):
    from .config import FlatIndexConfig, EnVectorConfig

    run(
        db=DBTYPE,
        db_config=EnVectorConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=FlatIndexConfig(),
        **parameters,
    )
