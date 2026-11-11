from importlib.metadata import version
from typing import Annotated, Unpack

import click

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run
from .. import DB
from .config import DuckDBConfig, DuckDBIndexConfig


class DuckDBTypedDict(CommonTypedDict):
    db_path: Annotated[
        str,
        click.option(
            "--db-path",
            type=click.Path(dir_okay=False),
            help="Path to a dedicated DuckDB benchmark database file.",
            required=True,
        ),
    ]
    threads: Annotated[
        int,
        click.option(
            "--threads",
            type=click.IntRange(min=1),
            default=1,
            help="Number of DuckDB threads used by each benchmark process.",
            show_default=True,
        ),
    ]


@cli.command(name="duckdb")
@click_parameter_decorators_from_typed_dict(DuckDBTypedDict)
def DuckDB(**parameters: Unpack[DuckDBTypedDict]) -> None:
    run(
        db=DB.DuckDB,
        db_config=DuckDBConfig(
            db_label=parameters["db_label"],
            version=version("duckdb"),
            db_path=parameters["db_path"],
            threads=parameters["threads"],
        ),
        db_case_config=DuckDBIndexConfig(),
        **parameters,
    )
