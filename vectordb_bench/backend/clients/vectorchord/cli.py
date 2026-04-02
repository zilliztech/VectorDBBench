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


class VectorChordTypedDict(CommonTypedDict):
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


class VectorChordRQTypedDict(VectorChordTypedDict):
    lists: Annotated[
        int | None,
        click.option(
            "--lists",
            type=int,
            help="Number of IVF lists for vchordrq index",
        ),
    ]
    probes: Annotated[
        int | None,
        click.option(
            "--probes",
            type=int,
            help="Number of probes during search",
            default=10,
            show_default=True,
        ),
    ]
    epsilon: Annotated[
        float | None,
        click.option(
            "--epsilon",
            type=float,
            help="Reranking precision factor (0.0-4.0, higher is more accurate but slower)",
            default=1.9,
            show_default=True,
        ),
    ]
    residual_quantization: Annotated[
        bool,
        click.option(
            "--residual-quantization/--no-residual-quantization",
            type=bool,
            help="Enable residual quantization for improved accuracy",
            default=False,
            show_default=True,
        ),
    ]
    spherical_centroids: Annotated[
        bool,
        click.option(
            "--spherical-centroids/--no-spherical-centroids",
            type=bool,
            help="L2-normalize centroids during K-means (recommended for cosine/IP)",
            default=False,
            show_default=True,
        ),
    ]
    build_threads: Annotated[
        int | None,
        click.option(
            "--build-threads",
            type=int,
            help="Number of threads for index building (range: 1-255)",
        ),
    ]
    degree_of_parallelism: Annotated[
        int | None,
        click.option(
            "--degree-of-parallelism",
            type=int,
            help="Degree of parallelism for index build (range: 1-256, default: 32)",
        ),
    ]
    max_scan_tuples: Annotated[
        int | None,
        click.option(
            "--max-scan-tuples",
            type=int,
            help="Max tuples to scan before stopping (-1 for unlimited)",
        ),
    ]
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel workers for index creation",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(VectorChordRQTypedDict)
def VectorChordRQ(
    **parameters: Unpack[VectorChordRQTypedDict],
):
    from .config import VectorChordConfig, VectorChordRQConfig

    run(
        db=DB.VectorChord,
        db_config=VectorChordConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=VectorChordRQConfig(
            lists=parameters["lists"],
            probes=parameters["probes"],
            epsilon=parameters["epsilon"],
            residual_quantization=parameters["residual_quantization"],
            spherical_centroids=parameters["spherical_centroids"],
            build_threads=parameters["build_threads"],
            degree_of_parallelism=parameters["degree_of_parallelism"],
            max_scan_tuples=parameters["max_scan_tuples"],
            max_parallel_workers=parameters["max_parallel_workers"],
        ),
        **parameters,
    )
