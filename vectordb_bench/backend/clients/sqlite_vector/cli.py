from importlib.metadata import version
from typing import Annotated, Unpack

import click

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run
from .. import DB
from .config import SQLiteVectorConfig, SQLiteVectorIndexConfig


class SQLiteVectorTypedDict(CommonTypedDict):
    db_path: Annotated[
        str,
        click.option(
            "--db-path",
            type=click.Path(dir_okay=False),
            help="Path to a dedicated SQLite-vector benchmark database file.",
            required=True,
        ),
    ]


@cli.command(name="sqlite-vector")
@click_parameter_decorators_from_typed_dict(SQLiteVectorTypedDict)
def SQLiteVector(**parameters: Unpack[SQLiteVectorTypedDict]) -> None:
    run(
        db=DB.SQLiteVector,
        db_config=SQLiteVectorConfig(
            db_label=parameters["db_label"],
            version=version("sqliteai-vector"),
            db_path=parameters["db_path"],
        ),
        db_case_config=SQLiteVectorIndexConfig(),
        **parameters,
    )
