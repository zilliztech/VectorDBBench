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

from ..milvus.cli import MilvusTypedDict

DBTYPE = DB.AliyunMilvus


class AliyunMilvusTypedDict(MilvusTypedDict):
    load_reqs_size: Annotated[
        int,
        click.option(
            "--load-reqs-size",
            type=int,
            help="Max request payload size (bytes) used to compute insert batch size",
            required=False,
            default=int(1.5 * 1024 * 1024),
            show_default=True,
        ),
    ]
    load_after_compaction: Annotated[
        bool,
        click.option(
            "--load-after-compaction/--no-load-after-compaction",
            "load_after_compaction",
            help=(
                "If True, defer load_collection until after compaction & index build in optimize(); "
                "if False (default), load_collection right after collection creation and call "
                "refresh_load at the end of optimize()."
            ),
            default=False,
            show_default=True,
        ),
    ]


class AliyunMilvusDISKANNTypedDict(CommonTypedDict, AliyunMilvusTypedDict):
    max_degree: Annotated[
        int,
        click.option(
            "--max-degree",
            type=int,
            help="R (max degree)",
            required=False,
            default=56,
            show_default=True,
        ),
    ]
    search_list: Annotated[
        int,
        click.option(
            "--search-list",
            type=int,
            help="L (search list size) used during DISKANN index search",
            required=False,
            default=200,
            show_default=True,
        ),
    ]
    build_search_list: Annotated[
        int,
        click.option(
            "--build-search-list",
            type=int,
            help="L (search list size) used during DISKANN index build",
            required=False,
            default=200,
            show_default=True,
        ),
    ]
    legacy: Annotated[
        bool,
        click.option(
            "--legacy/--no-legacy",
            "legacy",
            help="Use legacy Aliyun DISKANN behavior",
            default=False,
            show_default=True,
        ),
    ]
    store_strategy: Annotated[
        str,
        click.option(
            "--store-strategy",
            type=click.Choice(["MEMORY", "DISK"], case_sensitive=False),
            help="Aliyun store strategy",
            required=False,
            default="MEMORY",
            show_default=True,
        ),
    ]
    quant_type: Annotated[
        str,
        click.option(
            "--quant-type",
            type=click.Choice(["RABITQ", "PQ"], case_sensitive=False),
            help="Aliyun quant type",
            required=False,
            default="RABITQ",
            show_default=True,
        ),
    ]
    num_threads: Annotated[
        int,
        click.option(
            "--num-threads",
            type=int,
            help="Degree of parallelism",
            required=False,
            default=4,
            show_default=True,
        ),
    ]
    distance_strategy: Annotated[
        str,
        click.option(
            "--distance-strategy",
            type=click.Choice(
                ["FULL", "SINGLE QUANT", "QUANT THEN FULL", "QUANT THEN MORE BITS"],
                case_sensitive=False,
            ),
            help="Aliyun distance strategy",
            required=False,
            default="QUANT THEN MORE BITS",
            show_default=True,
        ),
    ]
    enable_prefetch: Annotated[
        bool,
        click.option(
            "--enable-prefetch/--disable-prefetch",
            "enable_prefetch",
            help="Enable Aliyun DISKANN prefetch during search",
            default=False,
            show_default=True,
        ),
    ]
    enable_thp: Annotated[
        bool,
        click.option(
            "--enable-thp/--disable-thp",
            "enable_thp",
            help="Enable Aliyun DISKANN transparent huge pages on collection load",
            default=False,
            show_default=True,
        ),
    ]
    rerank_topk_multiplier: Annotated[
        int | None,
        click.option(
            "--rerank-topk-multiplier",
            type=int,
            help=(
                "Search param: multiplier of topk for the rerank candidate budget "
                "(0 disables rerank read). Omit to NOT send this search param."
            ),
            required=False,
            default=None,
        ),
    ]
    early_termination_threshold: Annotated[
        int | None,
        click.option(
            "--early-termination-threshold",
            type=int,
            help="Search param: early termination threshold (0 disables). Omit to NOT send this search param.",
            required=False,
            default=None,
        ),
    ]
    cross_segment_rerank: Annotated[
        bool | None,
        click.option(
            "--cross-segment-rerank/--no-cross-segment-rerank",
            "cross_segment_rerank",
            help="Search param: enable cross-segment rerank in the reduce layer. Omit to NOT send this search param.",
            default=None,
        ),
    ]


@cli.command(name="aliyunmilvusdiskann")
@click_parameter_decorators_from_typed_dict(AliyunMilvusDISKANNTypedDict)
def AliyunMilvusDISKANN(**parameters: Unpack[AliyunMilvusDISKANNTypedDict]):
    from .config import AliyunMilvusConfig, AliyunMilvusDISKANNConfig

    run(
        db=DBTYPE,
        db_config=AliyunMilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=int(parameters["num_shards"]),
            replica_number=int(parameters["replica_number"]),
            load_reqs_size=int(parameters["load_reqs_size"]),
            load_after_compaction=bool(parameters["load_after_compaction"]),
        ),
        db_case_config=AliyunMilvusDISKANNConfig(
            search_list=parameters["search_list"],
            build_search_list=parameters["build_search_list"],
            max_degree=parameters["max_degree"],
            legacy=parameters["legacy"],
            store_strategy=parameters["store_strategy"],
            quant_type=parameters["quant_type"],
            num_threads=parameters["num_threads"],
            distance_strategy=parameters["distance_strategy"],
            enable_prefetch=bool(parameters["enable_prefetch"]),
            enable_thp=bool(parameters["enable_thp"]),
            # Pass through as-is; None means "not specified" -> omitted from search params.
            rerank_topk_multiplier=parameters["rerank_topk_multiplier"],
            early_termination_threshold=parameters["early_termination_threshold"],
            cross_segment_rerank=parameters["cross_segment_rerank"],
        ),
        **parameters,
    )
