import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    IVFFlatTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class PgVectoRSTypedDict(CommonTypedDict):
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
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel processes per maintenance operation (index creation)",
            required=False,
        ),
    ]
    quantization_type: Annotated[
        str,
        click.option(
            "--quantization-type",
            type=click.Choice(["trivial", "scalar", "product"]),
            help="quantization type for vectors",
            required=False,
        ),
    ]
    quantization_ratio: Annotated[
        str,
        click.option(
            "--quantization-ratio",
            type=click.Choice(["x4", "x8", "x16", "x32", "x64"]),
            help="quantization ratio(for product quantization)",
            required=False,
        ),
    ]


class PgVectoRSFlatTypedDict(PgVectoRSTypedDict, IVFFlatTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectoRSFlatTypedDict)
def PgVectoRSFlat(
    **parameters: Unpack[PgVectoRSFlatTypedDict],
):
    from .config import PgVectoRSConfig, PgVectoRSFLATConfig

    run(
        db=DB.PgVectoRS,
        db_config=PgVectoRSConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgVectoRSFLATConfig(
            max_parallel_workers=parameters["max_parallel_workers"],
            quantization_type=parameters["quantization_type"],
            quantization_ratio=parameters["quantization_ratio"],
        ),
        **parameters,
    )


class PgVectoRSIVFFlatTypedDict(PgVectoRSTypedDict, IVFFlatTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectoRSIVFFlatTypedDict)
def PgVectoRSIVFFlat(
    **parameters: Unpack[PgVectoRSIVFFlatTypedDict],
):
    from .config import PgVectoRSConfig, PgVectoRSIVFFlatConfig

    run(
        db=DB.PgVectoRS,
        db_config=PgVectoRSConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgVectoRSIVFFlatConfig(
            max_parallel_workers=parameters["max_parallel_workers"],
            quantization_type=parameters["quantization_type"],
            quantization_ratio=parameters["quantization_ratio"],
            probes=parameters["probes"],
            lists=parameters["lists"],
        ),
        **parameters,
    )


class PgVectoRSHNSWTypedDict(PgVectoRSTypedDict, HNSWFlavor1): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectoRSHNSWTypedDict)
def PgVectoRSHNSW(
    **parameters: Unpack[PgVectoRSHNSWTypedDict],
):
    from .config import PgVectoRSConfig, PgVectoRSHNSWConfig

    run(
        db=DB.PgVectoRS,
        db_config=PgVectoRSConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgVectoRSHNSWConfig(
            max_parallel_workers=parameters["max_parallel_workers"],
            quantization_type=parameters["quantization_type"],
            quantization_ratio=parameters["quantization_ratio"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
