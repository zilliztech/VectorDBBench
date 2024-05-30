from typing import Annotated, Optional, TypedDict, Unpack

import click
import os
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    IVFFlatTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from vectordb_bench.backend.clients import DB


class PgVectorTypedDict(CommonTypedDict):
    user_name: Annotated[
        str, click.option("--user-name", type=str, help="Db username", required=True)
    ]
    password: Annotated[
        str,
        click.option("--password",
                     type=str,
                     help="Postgres database password",
                     default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
                     show_default="$POSTGRES_PASSWORD",
                     ),
    ]

    host: Annotated[
        str, click.option("--host", type=str, help="Db host", required=True)
    ]
    db_name: Annotated[
        str, click.option("--db-name", type=str, help="Db name", required=True)
    ]
    maintenance_work_mem: Annotated[
        Optional[str],
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
        Optional[int],
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel processes per maintenance operation (index creation)",
            required=False,
        ),
    ]


class PgVectorIVFFlatTypedDict(PgVectorTypedDict, IVFFlatTypedDict):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectorIVFFlatTypedDict)
def PgVectorIVFFlat(
    **parameters: Unpack[PgVectorIVFFlatTypedDict],
):
    from .config import PgVectorConfig, PgVectorIVFFlatConfig

    run(
        db=DB.PgVector,
        db_config=PgVectorConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgVectorIVFFlatConfig(
            metric_type=None, lists=parameters["lists"], probes=parameters["probes"]
        ),
        **parameters,
    )


class PgVectorHNSWTypedDict(PgVectorTypedDict, HNSWFlavor1):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectorHNSWTypedDict)
def PgVectorHNSW(
    **parameters: Unpack[PgVectorHNSWTypedDict],
):
    from .config import PgVectorConfig, PgVectorHNSWConfig

    run(
        db=DB.PgVector,
        db_config=PgVectorConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgVectorHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
        ),
        **parameters,
    )
