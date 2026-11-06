from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class RaBitQLibTypedDict(CommonTypedDict):
    """Options shared by every rabitqlib index type."""

    path: Annotated[
        str,
        click.option("--path", type=str, help="Directory for the load buffer and the built index", required=True),
    ]
    seed: Annotated[
        int,
        click.option("--seed", type=int, default=100, show_default=True, help="Rotation and graph seed"),
    ]
    build_threads: Annotated[
        int,
        click.option(
            "--build-threads",
            type=click.IntRange(min=0),
            default=0,
            show_default=True,
            help="Build worker threads; 0 uses every CPU, capped by OMP_NUM_THREADS",
        ),
    ]


class RaBitQLibHNSWTypedDict(RaBitQLibTypedDict):
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=click.IntRange(1, 9),
            default=9,
            show_default=True,
            help="Total RaBitQ bits per coordinate, including the sign bit; 9 (sign + 8) is the fastest kernel",
        ),
    ]
    num_clusters: Annotated[
        int,
        click.option(
            "--num-clusters",
            type=click.IntRange(min=1),
            default=16,
            show_default=True,
            help="Centroids trained by the library over the whole dataset",
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=click.IntRange(min=2), default=16, show_default=True, help="HNSW graph degree M"),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=click.IntRange(min=1),
            default=200,
            show_default=True,
            help="HNSW construction search breadth",
        ),
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=click.IntRange(min=1),
            default=128,
            show_default=True,
            help="HNSW search breadth; raised to k when smaller",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(RaBitQLibHNSWTypedDict)
def RaBitQLibHNSW(**parameters: Unpack[RaBitQLibHNSWTypedDict]):
    from .config import RaBitQLibConfig, RaBitQLibHNSWConfig

    run(
        db=DB.RaBitQLib,
        db_config=RaBitQLibConfig(
            db_label=parameters["db_label"],
            path=parameters["path"],
        ),
        db_case_config=RaBitQLibHNSWConfig(
            nbits=parameters["nbits"],
            num_clusters=parameters["num_clusters"],
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            seed=parameters["seed"],
            build_threads=parameters["build_threads"],
        ),
        **parameters,
    )
