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
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel workers for index creation",
            required=False,
        ),
    ]
    quantization_type: Annotated[
        str | None,
        click.option(
            "--quantization-type",
            type=click.Choice(["vector", "halfvec", "rabitq8", "rabitq4"]),
            help="Quantization type for vectors",
            default="vector",
            show_default=True,
        ),
    ]


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
    rerank_in_table: Annotated[
        bool,
        click.option(
            "--rerank-in-table/--no-rerank-in-table",
            type=bool,
            help="Read vectors from table instead of storing in index (saves storage, degrades query performance)",
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
            quantization_type=parameters["quantization_type"],
            lists=parameters["lists"],
            probes=parameters["probes"],
            epsilon=parameters["epsilon"],
            residual_quantization=parameters["residual_quantization"],
            rerank_in_table=parameters["rerank_in_table"],
            spherical_centroids=parameters["spherical_centroids"],
            build_threads=parameters["build_threads"],
            degree_of_parallelism=parameters["degree_of_parallelism"],
            max_scan_tuples=parameters["max_scan_tuples"],
            max_parallel_workers=parameters["max_parallel_workers"],
        ),
        **parameters,
    )


class VectorChordGraphTypedDict(VectorChordTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="Max neighbors per vertex (default: 32)",
        ),
    ]
    ef_construction: Annotated[
        int | None,
        click.option(
            "--ef-construction",
            type=int,
            help="Dynamic list size during insertion (default: 64)",
        ),
    ]
    bits: Annotated[
        int | None,
        click.option(
            "--bits",
            type=int,
            help="RaBitQ quantization ratio (1 or 2, default: 2)",
        ),
    ]
    ef_search: Annotated[
        int | None,
        click.option(
            "--ef-search",
            type=int,
            help="Dynamic list size for search (default: 64)",
            default=64,
            show_default=True,
        ),
    ]
    beam_search: Annotated[
        int | None,
        click.option(
            "--beam-search",
            type=int,
            help="Batch vertex access width during search (default: 1)",
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


@cli.command()
@click_parameter_decorators_from_typed_dict(VectorChordGraphTypedDict)
def VectorChordGraph(
    **parameters: Unpack[VectorChordGraphTypedDict],
):
    from .config import VectorChordConfig, VectorChordGraphConfig

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
        db_case_config=VectorChordGraphConfig(
            quantization_type=parameters["quantization_type"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            bits=parameters["bits"],
            ef_search=parameters["ef_search"],
            beam_search=parameters["beam_search"],
            max_parallel_workers=parameters["max_parallel_workers"],
            max_scan_tuples=parameters["max_scan_tuples"],
        ),
        **parameters,
    )
