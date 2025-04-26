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
from ..api import IndexType


class LanceDBTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="URI connection string", required=True),
    ]
    token: Annotated[
        str | None,
        click.option("--token", type=str, help="Authentication token", required=False),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDB(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBConfig, _lancedb_case_config

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=_lancedb_case_config.get("NONE")(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDBAutoIndex(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBConfig, _lancedb_case_config

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=_lancedb_case_config.get(IndexType.AUTOINDEX)(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDBIVFPQ(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBConfig, _lancedb_case_config

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=_lancedb_case_config.get(IndexType.IVFPQ)(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDBHNSW(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBConfig, _lancedb_case_config

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=_lancedb_case_config.get(IndexType.HNSW)(),
        **parameters,
    )
