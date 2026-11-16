from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from ...cases import CaseLabel, CaseType, type2case
from .. import DB


def require_performance_case(case_type: str) -> None:
    """Refuse cases other than static performance before any data is written.

    The index is built once after loading, so capacity cases would only grow the
    load buffer, and streaming cases would search before an index exists.
    """
    label = type2case[CaseType[case_type]].model_fields["label"].default
    if label != CaseLabel.Performance:
        msg = (
            f"rabitqlib indexes support performance cases only, not {case_type} ({label.name}): "
            "the index is built once, after all vectors are loaded"
        )
        raise click.BadParameter(msg, param_hint="'--case-type'")


class RaBitQLibTypedDict(CommonTypedDict):
    """Options shared by every rabitqlib index type."""

    path: Annotated[
        str,
        click.option("--path", type=str, help="Directory for the load buffer and the built index", required=True),
    ]
    seed: Annotated[
        int,
        click.option(
            "--seed",
            type=int,
            default=100,
            show_default=True,
            help=(
                "Seed for the rotation, k-means and graph levels; the graph also depends on load order "
                "and build threads, so builds are not bit-for-bit reproducible"
            ),
        ),
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


# Not HNSWBaseTypedDict: its options have no defaults or ranges, and these mirror the
# library's defaults and the web UI's limits (dbCaseConfigs.py).
class RaBitQLibHNSWTypedDict(RaBitQLibTypedDict):
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=click.IntRange(1, 9),
            default=9,
            show_default=True,
            help="Total RaBitQ bits per coordinate, including the sign bit; 9 (sign + 8) is the highest precision",
        ),
    ]
    num_clusters: Annotated[
        int,
        click.option(
            "--num-clusters",
            type=click.IntRange(1, 65536),
            default=16,
            show_default=True,
            help="Centroids, trained with k-means on a sample of up to 256 x num-clusters vectors",
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=click.IntRange(2, 1024), default=16, show_default=True, help="HNSW graph degree M"),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=click.IntRange(1, 4096),
            default=200,
            show_default=True,
            help="HNSW construction search breadth",
        ),
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=click.IntRange(1, 65536),
            default=128,
            show_default=True,
            help="HNSW search breadth; raised to k when smaller",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(RaBitQLibHNSWTypedDict)
def RaBitQLibHNSW(**parameters: Unpack[RaBitQLibHNSWTypedDict]):
    from .config import RaBitQLibConfig, RaBitQLibHNSWConfig

    require_performance_case(parameters["case_type"])
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
