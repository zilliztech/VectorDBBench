from typing import Annotated, Unpack

import click
from click.core import ParameterSource

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB, MetricType
from .config import ZvecConfig, ZvecDiskANNIndexConfig, ZvecHNSWIndexConfig


class ZvecTypedDict(CommonTypedDict):
    path: Annotated[
        str,
        click.option("--path", type=str, help="collection path", required=True),
    ]


class ZvecHNSWTypedDict(CommonTypedDict, ZvecTypedDict):
    m: Annotated[
        int,
        click.option("--m", type=int, default=50, help="HNSW index parameter m."),
    ]
    ef_construction: Annotated[
        int,
        click.option("--ef-construction", type=int, default=500, help="HNSW index parameter ef_construction"),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, default=300, help="HNSW index parameter ef for search"),
    ]
    quantize_type: Annotated[
        str,
        click.option("--quantize-type", type=str, default="", help="HNSW index quantize type, fp16/int8 supported"),
    ]
    is_using_refiner: Annotated[
        bool,
        click.option(
            "--is-using-refiner",
            is_flag=True,
            default=False,
            help="is using refiner, suitable for quantized index, "
            "recall `ef-search` results then refine with unquantized vector to `topk` results",
        ),
    ]


class ZvecAlgorithmTypedDict(ZvecHNSWTypedDict):
    index_type: Annotated[
        str,
        click.option(
            "--index-type",
            type=click.Choice(["hnsw", "diskann"], case_sensitive=False),
            default="hnsw",
            show_default=True,
            help="Zvec vector index algorithm.",
        ),
    ]
    metric_type: Annotated[
        str | None,
        click.option(
            "--metric-type",
            type=click.Choice(["IP", "COSINE", "L2"], case_sensitive=False),
            default=None,
            help="Distance metric; omitted keeps the existing IP default.",
        ),
    ]
    max_degree: Annotated[
        int,
        click.option("--max-degree", type=click.IntRange(1, 100), default=64, help="DiskANN graph out-degree."),
    ]
    build_list_size: Annotated[
        int,
        click.option(
            "--build-list-size",
            type=click.IntRange(10, 100),
            default=100,
            help="DiskANN candidate list size during construction.",
        ),
    ]
    search_list_size: Annotated[
        int,
        click.option(
            "--search-list-size",
            type=click.IntRange(min=1),
            default=300,
            help="DiskANN query candidate list size; the engine uses at least top-k.",
        ),
    ]
    pq_chunk_num: Annotated[
        int,
        click.option(
            "--pq-chunk-num",
            type=click.IntRange(0, 1024),
            default=0,
            help="DiskANN PQ bytes per vector; 0 selects automatically, nonzero must not exceed dimension.",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(ZvecAlgorithmTypedDict)
def Zvec(**parameters: Unpack[ZvecAlgorithmTypedDict]):
    index_type = parameters["index_type"].lower()
    context = click.get_current_context()
    unsupported = (
        ("m", "ef_construction", "ef_search", "quantize_type", "is_using_refiner")
        if index_type == "diskann"
        else ("max_degree", "build_list_size", "search_list_size", "pq_chunk_num")
    )
    for parameter in unsupported:
        if context.get_parameter_source(parameter) not in (None, ParameterSource.DEFAULT):
            message = f"--{parameter.replace('_', '-')} does not apply to {index_type}"
            raise click.UsageError(message)

    metric_type = parameters["metric_type"]
    metric_type = MetricType(metric_type.upper()) if metric_type else None
    if index_type == "diskann":
        case_config = ZvecDiskANNIndexConfig(
            metric_type=metric_type,
            max_degree=parameters["max_degree"],
            build_list_size=parameters["build_list_size"],
            search_list_size=parameters["search_list_size"],
            pq_chunk_num=parameters["pq_chunk_num"],
        )
    else:
        case_config = ZvecHNSWIndexConfig(
            metric_type=metric_type,
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            quantize_type=parameters["quantize_type"],
            is_using_refiner=parameters["is_using_refiner"],
        )

    run(
        db=DB.Zvec,
        db_config=ZvecConfig(
            db_label=parameters["db_label"],
            path=parameters["path"],
        ),
        db_case_config=case_config,
        **parameters,
    )
