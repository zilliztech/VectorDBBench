from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    HNSWFlavor1,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from ..pgvector.cli import PgVectorTypedDict
from .config import DEFAULT_GRAPH_CACHE_TIMEOUT_SECONDS


def parse_quantization_nbits(ctx: object, param: object, value: str | None) -> int | None:  # noqa: ARG001
    return int(value) if value is not None else None


class PolarDBPgHNSWOptions(TypedDict):
    iterative_scan: Annotated[
        str,
        click.option(
            "--iterative-scan",
            type=click.Choice(["off", "strict_order", "relaxed_order"]),
            default="off",
            show_default=True,
            help="HNSW iterative scan mode; Graph Cache requires off",
        ),
    ]
    create_index_before_load: Annotated[
        bool,
        click.option(
            "--create-index-before-load/--skip-create-index-before-load",
            default=False,
            show_default=True,
            help="Create the HNSW index before loading data",
        ),
    ]
    create_index_after_load: Annotated[
        bool,
        click.option(
            "--create-index-after-load/--skip-create-index-after-load",
            default=True,
            show_default=True,
            help="Create the HNSW index after loading data",
        ),
    ]
    graph_cache: Annotated[
        bool,
        click.option(
            "--graph-cache/--skip-graph-cache",
            default=True,
            show_default=True,
            help="Build Graph Cache and wait until it is usable before search",
        ),
    ]
    graph_cache_timeout: Annotated[
        int,
        click.option(
            "--graph-cache-timeout",
            type=click.IntRange(min=1),
            default=DEFAULT_GRAPH_CACHE_TIMEOUT_SECONDS,
            show_default=True,
            help="Seconds to wait for Graph Cache to become usable",
        ),
    ]
    quantization: Annotated[
        str | None,
        click.option(
            "--quantization",
            type=click.Choice(["pq", "sq4", "sq8", "rabitq"]),
            help="PolarDB HNSW internal quantization method",
        ),
    ]
    pq_m: Annotated[
        int | None,
        click.option("--pq-m", type=int, help="Number of PQ sub-quantizers"),
    ]
    train_samples: Annotated[
        int | None,
        click.option("--train-samples", type=int, help="Number of quantizer training samples"),
    ]
    quantization_nbits: Annotated[
        int | None,
        click.option(
            "--quantization-nbits",
            type=click.Choice(["1", "4", "8"]),
            callback=parse_quantization_nbits,
            help="RaBitQ bits per dimension",
        ),
    ]


class PolarDBPgHNSWTypedDict(PgVectorTypedDict, HNSWFlavor1, PolarDBPgHNSWOptions): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PolarDBPgHNSWTypedDict)
def PolarDBPgHNSW(**parameters: Unpack[PolarDBPgHNSWTypedDict]):
    from .config import PolarDBPgConfig, PolarDBPgHNSWConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.PolarDBPG,
        db_config=PolarDBPgConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PolarDBPgHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            quantization_type=parameters["quantization_type"],
            table_quantization_type=parameters["table_quantization_type"],
            reranking=parameters["reranking"],
            reranking_metric=parameters["reranking_metric"],
            quantized_fetch_limit=parameters["quantized_fetch_limit"],
            iterative_scan=parameters["iterative_scan"],
            create_index_before_load=parameters["create_index_before_load"],
            create_index_after_load=parameters["create_index_after_load"],
            graph_cache=parameters["graph_cache"],
            graph_cache_timeout=parameters["graph_cache_timeout"],
            quantization=parameters["quantization"],
            pq_m=parameters["pq_m"],
            train_samples=parameters["train_samples"],
            quantization_nbits=parameters["quantization_nbits"],
        ),
        **parameters,
    )
