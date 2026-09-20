import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class OracleCommonTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--user-name",
            type=str,
            help="Oracle username",
            default=lambda: os.environ.get("ORACLE_USER", "vdbbench_user"),
            show_default="vdbbench_user",
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Oracle password",
            default=lambda: os.environ.get("ORACLE_PASSWORD", "vdbbench_pass"),
            show_default="$ORACLE_PASSWORD",
        ),
    ]
    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="Oracle host",
            default="localhost",
            show_default=True,
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Oracle port",
            default=1521,
            show_default=True,
        ),
    ]
    service_name: Annotated[
        str,
        click.option(
            "--service-name",
            type=str,
            help="Oracle Service Name",
            default="FREEPDB1",
            show_default=True,
        ),
    ]
    sysdba: Annotated[
        bool,
        click.option(
            "--sysdba",
            is_flag=True,
            default=False,
            help="Connect as sysdba role",
        ),
    ]


class OracleHNSWTypedDict(OracleCommonTypedDict):
    neighbors: Annotated[
        int,
        click.option(
            "--neighbors",
            type=int,
            default=32,
            show_default=True,
            help="NEIGHBORS for HNSW graph index",
        ),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            default=200,
            show_default=True,
            help="EFCONSTRUCTION for HNSW graph index",
        ),
    ]
    index_target_accuracy: Annotated[
        int,
        click.option(
            "--index-target-accuracy",
            type=int,
            default=95,
            show_default=True,
            help="Index-time TARGET ACCURACY (1-100)",
        ),
    ]
    search_target_accuracy: Annotated[
        int,
        click.option(
            "--search-target-accuracy",
            type=int,
            default=95,
            show_default=True,
            help="Search-time TARGET ACCURACY (1-100)",
        ),
    ]


class OracleIVFTypedDict(OracleCommonTypedDict):
    neighbor_partitions: Annotated[
        int,
        click.option(
            "--neighbor-partitions",
            type=int,
            default=1024,
            show_default=True,
            help="NEIGHBOR PARTITIONS for IVF index",
        ),
    ]
    samples_per_partition: Annotated[
        int,
        click.option(
            "--samples-per-partition",
            type=int,
            default=10,
            show_default=True,
            help="SAMPLES_PER_PARTITION for IVF index",
        ),
    ]
    min_vectors_per_partition: Annotated[
        int,
        click.option(
            "--min-vectors-per-partition",
            type=int,
            default=5,
            show_default=True,
            help="MIN_VECTORS_PER_PARTITION for IVF index",
        ),
    ]
    index_target_accuracy: Annotated[
        int,
        click.option(
            "--index-target-accuracy",
            type=int,
            default=90,
            show_default=True,
            help="Index-time TARGET ACCURACY (1-100)",
        ),
    ]
    search_target_accuracy: Annotated[
        int,
        click.option(
            "--search-target-accuracy",
            type=int,
            default=90,
            show_default=True,
            help="Search-time TARGET ACCURACY (1-100)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(OracleCommonTypedDict)
def oracleflat(**parameters: Unpack[OracleCommonTypedDict]):
    from .config import OracleConfig, OracleFlatConfig

    run(
        db=DB.Oracle,
        db_config=OracleConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            service_name=parameters["service_name"],
            sysdba=parameters["sysdba"],
        ),
        db_case_config=OracleFlatConfig(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(OracleHNSWTypedDict)
def oraclehnsw(**parameters: Unpack[OracleHNSWTypedDict]):
    from .config import OracleConfig, OracleHNSWConfig

    run(
        db=DB.Oracle,
        db_config=OracleConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            service_name=parameters["service_name"],
            sysdba=parameters["sysdba"],
        ),
        db_case_config=OracleHNSWConfig(
            neighbors=parameters["neighbors"],
            ef_construction=parameters["ef_construction"],
            index_target_accuracy=parameters["index_target_accuracy"],
            search_target_accuracy=parameters["search_target_accuracy"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(OracleIVFTypedDict)
def oracleivf(**parameters: Unpack[OracleIVFTypedDict]):
    from .config import OracleConfig, OracleIVFConfig

    run(
        db=DB.Oracle,
        db_config=OracleConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            service_name=parameters["service_name"],
            sysdba=parameters["sysdba"],
        ),
        db_case_config=OracleIVFConfig(
            neighbor_partitions=parameters["neighbor_partitions"],
            samples_per_partition=parameters["samples_per_partition"],
            min_vectors_per_partition=parameters["min_vectors_per_partition"],
            index_target_accuracy=parameters["index_target_accuracy"],
            search_target_accuracy=parameters["search_target_accuracy"],
        ),
        **parameters,
    )
