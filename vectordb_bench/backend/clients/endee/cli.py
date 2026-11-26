from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    benchmark_runner,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

from ..api import EmptyDBCaseConfig
from .config import EndeeConfig

# `benchmark_runner` is re-exported for backward compatibility: the previous Endee CLI
# imported it here, and existing callers/tests patch `endee.cli.benchmark_runner`.
__all__ = ["Endee", "EndeeTypedDict", "benchmark_runner"]


class EndeeTypedDict(CommonTypedDict):
    token: Annotated[str, click.option("--token", type=str, required=True, default=None, help="Endee API token")]
    base_url: Annotated[
        str | None,
        click.option(
            "--base-url",
            type=str,
            default=None,
            help="API server URL. Leave unset to auto-detect from a serverless token "
            "or the local self-hosted default.",
        ),
    ]
    space_type: Annotated[
        str,
        click.option(
            "--space-type",
            type=click.Choice(["cosine", "l2", "ip"]),
            default="cosine",
            help="Distance metric",
            show_default=True,
        ),
    ]
    precision: Annotated[
        str,
        click.option(
            "--precision",
            type=click.Choice(["binary", "int8", "int8e", "int16", "float16", "float32"]),
            default="int16",
            help="Quant Level",
            show_default=True,
        ),
    ]
    version: Annotated[
        str,
        click.option(
            "--version",
            type=str,
            default="",
            help="Endee server/SDK version label recorded in the results (metadata only)",
        ),
    ]
    m: Annotated[int, click.option("--m", type=int, default=16, help="HNSW M parameter", show_default=True)]
    ef_con: Annotated[
        int, click.option("--ef-con", type=int, default=128, help="HNSW construction parameter", show_default=True)
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=click.IntRange(1, 1024),
            default=128,
            help="HNSW search parameter",
            show_default=True,
        ),
    ]
    collection_name: Annotated[
        str,
        click.option(
            "--collection-name",
            "--index-name",
            "collection_name",
            type=str,
            required=True,
            help="Endee collection name (--index-name is a deprecated alias)",
        ),
    ]
    prefilter_cardinality_threshold: Annotated[
        int,
        click.option(
            "--prefilter-cardinality-threshold",
            type=click.IntRange(1_000, 1_000_000),
            default=10_000,
            help="Filtered search only: use brute-force prefiltering when the filter matches at most N vectors.",
            show_default=True,
        ),
    ]
    filter_boost_percentage: Annotated[
        int,
        click.option(
            "--filter-boost-percentage",
            type=click.IntRange(0, 100),
            default=0,
            help="Filtered search only: expand the HNSW candidate pool by this percentage before filtering.",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(EndeeTypedDict)
def Endee(**parameters: Unpack[EndeeTypedDict]):
    """
    Run VectorDBBench against Endee VectorDB (collections-based API).

    Python Package: pip install "endee>=2.2.0"
    """
    params_for_nd = {k: v for k, v in parameters.items() if v is not None}
    run(
        db=DB.Endee,
        db_config=EndeeConfig(**params_for_nd),
        db_case_config=EmptyDBCaseConfig(),
        **parameters,
    )
