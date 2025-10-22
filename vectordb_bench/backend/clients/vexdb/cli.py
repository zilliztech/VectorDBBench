import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
# from vectordb_bench.backend.clients.api import MetricType

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    IVFFlatTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class VexDBTypedDict(CommonTypedDict):
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
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
            required=False,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
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
    create_index_before_load: Annotated[
        bool | None,
        click.option(
            "--create-index-before-load",
            type=bool,
            help="Whether create index before load,Streaming case recommended to be true，default is false",
            required=False,
            default=False,
        ),
    ]


class VexDBIVFFlatTypedDict(VexDBTypedDict, IVFFlatTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(VexDBIVFFlatTypedDict)
def VexDBIVFFlat(
    **parameters: Unpack[VexDBIVFFlatTypedDict],
):
    from .config import VexDBConfig, VexDBIVFFlatConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.VexDB,
        db_config=VexDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=VexDBIVFFlatConfig(
            lists=parameters["lists"],
            probes=parameters["probes"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            create_index_before_load=parameters["create_index_before_load"],
        ),
        **parameters,
    )


class VexDBHNSWTypedDict(VexDBTypedDict, HNSWFlavor1): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(VexDBHNSWTypedDict)
def VexDBHNSW(
    **parameters: Unpack[VexDBHNSWTypedDict],
):
    from .config import VexDBConfig, VexDBHNSWConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.VexDB,
        db_config=VexDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=VexDBHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            create_index_before_load=parameters["create_index_before_load"],
        ),
        **parameters,
    )
