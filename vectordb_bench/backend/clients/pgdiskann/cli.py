import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class PgDiskAnnTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Postgres database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]

    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
    max_neighbors: Annotated[
        int,
        click.option(
            "--max-neighbors",
            type=int,
            help="PgDiskAnn max neighbors",
        ),
    ]
    l_value_ib: Annotated[
        int,
        click.option(
            "--l-value-ib",
            type=int,
            help="PgDiskAnn l_value_ib",
        ),
    ]
    l_value_is: Annotated[
        float,
        click.option(
            "--l-value-is",
            type=float,
            help="PgDiskAnn l_value_is",
        ),
    ]
    maintenance_work_mem: Annotated[
        str | None,
        click.option(
            "--maintenance-work-mem",
            type=str,
            help="Sets the maximum memory to be used for maintenance operations (index creation). "
            "Can be entered as string with unit like '64GB' or as an integer number of KB."
            "This will set the parameters: max_parallel_maintenance_workers,"
            " max_parallel_workers & table(parallel_workers)",
            required=False,
        ),
    ]
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel processes per maintenance operation (index creation)",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(PgDiskAnnTypedDict)
def PgDiskAnn(
    **parameters: Unpack[PgDiskAnnTypedDict],
):
    from .config import PgDiskANNConfig, PgDiskANNImplConfig

    run(
        db=DB.PgDiskANN,
        db_config=PgDiskANNConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgDiskANNImplConfig(
            max_neighbors=parameters["max_neighbors"],
            l_value_ib=parameters["l_value_ib"],
            l_value_is=parameters["l_value_is"],
            max_parallel_workers=parameters["max_parallel_workers"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
        ),
        **parameters,
    )
