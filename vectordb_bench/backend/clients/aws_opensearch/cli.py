import logging
from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from .config import AWSOS_Engine, AWSOSQuantization

log = logging.getLogger(__name__)


class AWSOpenSearchTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=80, help="Db Port")]
    user: Annotated[str, click.option("--user", type=str, help="Db User")]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    number_of_shards: Annotated[
        int,
        click.option("--number-of-shards", type=int, help="Number of primary shards for the index", default=1),
    ]
    number_of_replicas: Annotated[
        int,
        click.option(
            "--number-of-replicas", type=int, help="Number of replica copies for each primary shard", default=1
        ),
    ]
    index_thread_qty: Annotated[
        int,
        click.option(
            "--index-thread-qty",
            type=int,
            help="Thread count for native engine indexing",
            default=4,
        ),
    ]

    engine: Annotated[
        str,
        click.option(
            "--engine",
            type=click.Choice(["faiss", "lucene", "s3vector"], case_sensitive=False),
            help="HNSW algorithm implementation to use",
            default="faiss",
        ),
    ]

    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice(["l2", "cosine", "ip"], case_sensitive=False),
            help="Distance metric type for vector similarity",
            default="l2",
        ),
    ]

    number_of_segments: Annotated[
        int,
        click.option("--number-of-segments", type=int, help="Target number of segments after merging", default=1),
    ]

    refresh_interval: Annotated[
        str,
        click.option(
            "--refresh-interval", type=str, help="How often to make new data available for search", default="60s"
        ),
    ]

    force_merge_enabled: Annotated[
        bool,
        click.option("--force-merge-enabled", type=bool, help="Whether to perform force merge operation", default=True),
    ]

    flush_threshold_size: Annotated[
        str,
        click.option(
            "--flush-threshold-size", type=str, help="Size threshold for flushing the transaction log", default="5120mb"
        ),
    ]

    cb_threshold: Annotated[
        str,
        click.option(
            "--cb-threshold",
            type=str,
            help="k-NN Memory circuit breaker threshold",
            default="50%",
        ),
    ]

    index_thread_qty_during_force_merge: Annotated[
        int,
        click.option(
            "--index-thread-qty-during-force-merge",
            type=int,
            help="Thread count during force merge operations",
            default=8,
        ),
    ]

    number_of_indexing_clients: Annotated[
        int,
        click.option(
            "--number-of-indexing-clients",
            type=int,
            help="Number of concurrent indexing clients",
            default=1,
        ),
    ]

    ef_construction: Annotated[
        int | None,
        click.option(
            "--ef-construction",
            type=int,
            help="ef parameter for HNSW construction (not used for s3vector engine)",
            default=None,
            required=False,
        ),
    ]

    quantization_type: Annotated[
        str | None,
        click.option(
            "--quantization-type",
            type=click.Choice(["fp32", "fp16"]),
            help="quantization type for vectors (in index)",
            default="fp32",
            required=False,
        ),
    ]


class AWSOpenSearchHNSWTypedDict(CommonTypedDict, AWSOpenSearchTypedDict, HNSWFlavor1): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AWSOpenSearchHNSWTypedDict)
def AWSOpenSearch(**parameters: Unpack[AWSOpenSearchHNSWTypedDict]):
    from .config import AWSOpenSearchConfig, AWSOpenSearchIndexConfig

    # Set default values for HNSW parameters if not provided and not using s3vector
    engine = AWSOS_Engine(parameters["engine"])
    ef_construction = parameters.get("ef_construction")
    ef_search = parameters.get("ef_search")
    m = parameters.get("m")

    # For non-s3vector engines, provide defaults if None
    if engine != AWSOS_Engine.s3vector:
        if ef_construction is None:
            ef_construction = 200
        if ef_search is None:
            ef_search = 100
        if m is None:
            m = 16

    run(
        db=DB.AWSOpenSearch,
        db_config=AWSOpenSearchConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=AWSOpenSearchIndexConfig(
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            index_thread_qty=parameters["index_thread_qty"],
            number_of_segments=parameters["number_of_segments"],
            refresh_interval=parameters["refresh_interval"],
            force_merge_enabled=parameters["force_merge_enabled"],
            flush_threshold_size=parameters["flush_threshold_size"],
            index_thread_qty_during_force_merge=parameters["index_thread_qty_during_force_merge"],
            number_of_indexing_clients=parameters["number_of_indexing_clients"],
            cb_threshold=parameters["cb_threshold"],
            efConstruction=ef_construction,
            ef_search=ef_search,
            M=m,
            engine=engine,
            quantization_type=AWSOSQuantization(parameters["quantization_type"]),
            metric_type_name=parameters["metric_type"],
        ),
        **parameters,
    )
