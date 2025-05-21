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

log = logging.getLogger(__name__)


class AWSOpenSearchTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=443, help="Db Port")]
    user: Annotated[str, click.option("--user", type=str, default="admin", help="Db User")]
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
            type=click.Choice(["nmslib", "faiss", "lucene"], case_sensitive=False),
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

    number_of_indexing_clients: Annotated[
        int,
        click.option(
            "--number-of-indexing-clients",
            type=int,
            help="Number of concurrent indexing clients",
            default=1,
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
            default=4,
        ),
    ]

    faiss_use_fp16: Annotated[
        bool,
        click.option(
            "--faiss-use-fp16/--no-faiss-use-fp16",
            default=True,
            help="Whether to use fp16 encoder for faiss engine",
        ),
    ]


class AWSOpenSearchHNSWTypedDict(CommonTypedDict, AWSOpenSearchTypedDict, HNSWFlavor1): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AWSOpenSearchHNSWTypedDict)
def AWSOpenSearch(**parameters: Unpack[AWSOpenSearchHNSWTypedDict]):
    from vectordb_bench.backend.clients.api import MetricType

    from .config import AWSOpenSearchConfig, AWSOpenSearchIndexConfig, AWSOS_Engine

    log.info(f"CLI parameters: {parameters}")

    ef_search = parameters.get("ef_search", 256)
    ef_construction = parameters.get("ef_construction", 256)
    m = parameters.get("m", 16)

    engine_name = parameters.get("engine", "faiss")
    metric_type_name = parameters.get("metric_type", "l2")

    log.info(f"ef_search from CLI: {ef_search}")
    log.info(f"engine from CLI: {engine_name}")
    log.info(f"metric_type from CLI: {metric_type_name}")

    run(
        db=DB.AWSOpenSearch,
        db_config=AWSOpenSearchConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
            db_label=parameters.get("db_label", ""),
        ),
        db_case_config=AWSOpenSearchIndexConfig(
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            index_thread_qty=parameters["index_thread_qty"],
            number_of_segments=parameters["number_of_segments"],
            refresh_interval=parameters["refresh_interval"],
            force_merge_enabled=parameters["force_merge_enabled"],
            flush_threshold_size=parameters["flush_threshold_size"],
            number_of_indexing_clients=parameters["number_of_indexing_clients"],
            index_thread_qty_during_force_merge=parameters["index_thread_qty_during_force_merge"],
            cb_threshold=parameters["cb_threshold"],
            ef_search=ef_search,
            efConstruction=ef_construction,
            M=m,
            engine_name=engine_name,
            metric_type_name=metric_type_name,
            metric_type=MetricType[parameters["metric_type"].upper()],
            faiss_use_fp16=(
                parameters.get("faiss_use_fp16", True) if AWSOS_Engine[engine_name] == AWSOS_Engine.faiss else False
            ),
        ),
        **parameters,
    )
