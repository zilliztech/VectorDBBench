#!/usr/bin/env python3
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


class VolcMySQLTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            required=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Password",
            required=True,
        ),
    ]

    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="Db host",
            default="127.0.0.1",
        ),
    ]

    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=3306,
            help="DB Port",
        ),
    ]


class VolcMySQLHNSWTypedDict(VolcMySQLTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="M parameter in HNSW vector indexing",
            required=False,
        ),
    ]

    ef_search: Annotated[
        int | None,
        click.option(
            "--ef-search",
            type=int,
            help="Session variable loose_hnsw_ef_search",
            required=False,
        ),
    ]

    ef_construction: Annotated[
        int | None,
        click.option(
            "--ef-construction",
            type=int,
            help="HNSW ef_construction",
            required=False,
        ),
    ]

    quant_algorithm: Annotated[
        str | None,
        click.option(
            "--quant-algorithm",
            type=click.Choice(["NONE", "SQ", "PQ"]),
            help="Quantization algorithm",
            required=False,
        ),
    ]

    quant_type: Annotated[
        str | None,
        click.option(
            "--quant-type",
            type=click.Choice(["16_bit", "8_bit", "4_bit", "binary"]),
            help="Quantization type",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(VolcMySQLHNSWTypedDict)
def VolcMySQLHNSW(
    **parameters: Unpack[VolcMySQLHNSWTypedDict],
):
    from .config import VolcMySQLConfig, VolcMySQLHNSWConfig

    run(
        db=DB.VolcMySQL,
        db_config=VolcMySQLConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=VolcMySQLHNSWConfig(
            M=parameters["m"],
            ef_search=parameters["ef_search"],
            ef_construction=parameters["ef_construction"],
            quant_algorithm=parameters["quant_algorithm"],
            quant_type=parameters["quant_type"],
        ),
        **parameters,
    )
