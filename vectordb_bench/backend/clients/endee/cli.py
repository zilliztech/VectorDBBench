from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

from ..api import EmptyDBCaseConfig
from .config import EndeeConfig, EndeeOSSConfig


class EndeeTypedDict(CommonTypedDict):
    token: Annotated[str, click.option("--token", type=str, required=True, default=None, help="Endee API token")]
    region: Annotated[str, click.option("--region", type=str, default=None, help="Endee region", show_default=True)]
    base_url: Annotated[
        str,
        click.option(
            "--base-url", type=str, default="http://127.0.0.1:8080/api/v2", help="API server URL", show_default=True
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
    version: Annotated[int, click.option("--version", type=str, default=None, help="Index version", show_default=True)]
    m: Annotated[int, click.option("--m", type=int, default=None, help="HNSW M parameter", show_default=True)]
    ef_con: Annotated[
        int, click.option("--ef-con", type=int, default=None, help="HNSW construction parameter", show_default=True)
    ]
    ef_search: Annotated[
        int, click.option("--ef-search", type=int, default=None, help="HNSW search parameter", show_default=True)
    ]
    collection_name: Annotated[
        str,
        click.option("--collection-name", type=str, required=True, help="Endee collection name"),
    ]
    prefilter_cardinality_threshold: Annotated[
        int,
        click.option(
            "--prefilter-cardinality-threshold",
            type=int,
            default=None,
            help="Use brute-force prefiltering when filter matches ≤N vectors (1k-1M, default: 10k).",
            show_default=True,
        ),
    ]
    filter_boost_percentage: Annotated[
        int,
        click.option(
            "--filter-boost-percentage",
            type=int,
            default=None,
            help="Increase search limit to offset filtered-out results (range: 0-100, default: 0).",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(EndeeTypedDict)
def Endee(**parameters: Unpack[EndeeTypedDict]):
    """
    Run VectorDBBench against Endee VectorDB (collections-based API, v2).
    """
    params_for_nd = {k: v for k, v in parameters.items() if v is not None}
    run(
        db=DB.Endee,
        db_config=EndeeConfig(**params_for_nd),
        db_case_config=EmptyDBCaseConfig(),
        **parameters,
    )


# Endee OSS (v1, index-based API)
# Python Package: pip install endee==1.0.0
# Docs: https://docs.endee.io/v1/overview
# OSS Repo: https://github.com/endee-io/endee
class EndeeOSSTypedDict(CommonTypedDict):
    token: Annotated[str, click.option("--token", type=str, required=True, default=None, help="Endee API token")]
    region: Annotated[str, click.option("--region", type=str, default=None, help="Endee region", show_default=True)]
    base_url: Annotated[
        str,
        click.option(
            "--base-url", type=str, default="http://127.0.0.1:8080/api/v1", help="API server URL", show_default=True
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
            type=click.Choice(["binary", "int8", "int16", "float16", "float32"]),
            default="int8",
            help="Quant Level",
            show_default=True,
        ),
    ]
    version: Annotated[int, click.option("--version", type=int, default=None, help="Index version", show_default=True)]
    m: Annotated[int, click.option("--m", type=int, default=None, help="HNSW M parameter", show_default=True)]
    ef_con: Annotated[
        int, click.option("--ef-con", type=int, default=None, help="HNSW construction parameter", show_default=True)
    ]
    ef_search: Annotated[
        int, click.option("--ef-search", type=int, default=None, help="HNSW search parameter", show_default=True)
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", type=str, required=True, help="Endee index name"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(EndeeOSSTypedDict)
def EndeeOSS(**parameters: Unpack[EndeeOSSTypedDict]):
    """
    Run VectorDBBench against Endee OSS (v1, index-based API).

    Python Package: pip install endee==1.0.0
    Docs: https://docs.endee.io/v1/overview
    OSS Repo: https://github.com/endee-io/endee
    """
    params_for_nd = {k: v for k, v in parameters.items() if v is not None}
    run(
        db=DB.EndeeOSS,
        db_config=EndeeOSSConfig(**params_for_nd),
        db_case_config=EmptyDBCaseConfig(),
        **parameters,
    )
