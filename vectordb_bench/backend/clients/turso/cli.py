from importlib.metadata import version
from typing import Annotated, Unpack

import click

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run
from .. import DB
from .config import TursoConfig, TursoIndexConfig


class TursoTypedDict(CommonTypedDict):
    db_path: Annotated[
        str,
        click.option(
            "--db-path",
            type=click.Path(dir_okay=False),
            help="Path to a dedicated Turso benchmark database file.",
            required=True,
        ),
    ]
    experimental_multiprocess_wal: Annotated[
        bool,
        click.option(
            "--experimental-multiprocess-wal/--no-experimental-multiprocess-wal",
            default=True,
            help="Enable Turso's experimental multi-process WAL for concurrent benchmark processes.",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TursoTypedDict)
def Turso(**parameters: Unpack[TursoTypedDict]) -> None:
    run(
        db=DB.Turso,
        db_config=TursoConfig(
            db_label=parameters["db_label"],
            version=version("pyturso"),
            db_path=parameters["db_path"],
            experimental_multiprocess_wal=parameters["experimental_multiprocess_wal"],
        ),
        db_case_config=TursoIndexConfig(),
        **parameters,
    )
