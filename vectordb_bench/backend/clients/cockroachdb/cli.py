"""CLI parameter definitions for CockroachDB."""

from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class CockroachDBTypedDict(CommonTypedDict):
    """Type definition for CockroachDB CLI parameters."""

    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="CockroachDB username", default="root", show_default=True),
    ]
    password: Annotated[
        str,
        click.option("--password", type=str, help="CockroachDB password", default="", show_default=False),
    ]
    host: Annotated[
        str,
        click.option("--host", type=str, help="CockroachDB host", required=True),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="CockroachDB port", default=26257, show_default=True),
    ]
    db_name: Annotated[
        str,
        click.option("--db-name", type=str, help="Database name", required=True),
    ]
    sslmode: Annotated[
        str,
        click.option(
            "--sslmode",
            type=str,
            help="SSL mode (disable, require, verify-ca, verify-full)",
            default="disable",
            show_default=True,
        ),
    ]
    sslrootcert: Annotated[
        str | None,
        click.option(
            "--sslrootcert",
            type=str,
            help="Path to SSL root certificate (required for verify-ca, verify-full)",
            default=None,
        ),
    ]
    min_partition_size: Annotated[
        int | None,
        click.option(
            "--min-partition-size",
            type=int,
            help="Minimum vectors per partition (default: 16, range: 1-1024)",
            default=16,
            show_default=True,
        ),
    ]
    max_partition_size: Annotated[
        int | None,
        click.option(
            "--max-partition-size",
            type=int,
            help="Maximum vectors per partition (default: 128, range: 4x min-4096)",
            default=128,
            show_default=True,
        ),
    ]
    vector_search_beam_size: Annotated[
        int | None,
        click.option(
            "--vector-search-beam-size",
            type=int,
            help="Partitions explored during search (default: 32)",
            default=32,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(CockroachDBTypedDict)
def CockroachDB(
    **parameters: Unpack[CockroachDBTypedDict],
):
    """Run CockroachDB vector benchmark."""
    from .config import CockroachDBConfig, CockroachDBVectorIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)

    from vectordb_bench.backend.clients.api import MetricType

    # Use provided metric_type or default to COSINE
    metric_type = parameters.get("metric_type")
    if metric_type is None:
        metric_type = MetricType.COSINE
    elif isinstance(metric_type, str):
        metric_type = MetricType(metric_type)

    run(
        db=DB.CockroachDB,
        db_config=CockroachDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            sslmode=parameters.get("sslmode", "disable"),
            sslrootcert=parameters.get("sslrootcert"),
        ),
        db_case_config=CockroachDBVectorIndexConfig(
            metric_type=metric_type,
            min_partition_size=parameters.get("min_partition_size", 16),
            max_partition_size=parameters.get("max_partition_size", 128),
            vector_search_beam_size=parameters.get("vector_search_beam_size", 32),
        ),
        **parameters,
    )
