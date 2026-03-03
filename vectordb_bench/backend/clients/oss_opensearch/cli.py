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
from .config import OSSOpenSearchQuantization, OSSOS_Engine

log = logging.getLogger(__name__)


class OSSOpenSearchTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=80, help="Db Port")]
    user: Annotated[str, click.option("--user", type=str, help="Db User")]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    use_ssl: Annotated[
        bool | None,
        click.option("--use-ssl", type=bool, default=None, help="Use SSL (defaults to True when port=443)", required=False),
    ]
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

    index_thread_qty_during_force_merge: Annotated[
        int,
        click.option(
            "--index_thread_qty_during_force_merge",
            type=int,
            help="Thread count for native engine indexing during force merge",
            default=4,
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

    quantization_type: Annotated[
        str | None,
        click.option(
            "--quantization-type",
            type=click.Choice(["None", "LuceneSQ", "FaissSQfp16"]),
            help="quantization type for vectors (in index)",
            default="None",
            required=False,
        ),
    ]

    confidence_interval: Annotated[
        float | None,
        click.option(
            "--confidence-interval",
            type=float,
            help="Confidence interval for Lucene SQ (0.0-1.0, optional)",
            default=None,
            required=False,
        ),
    ]

    clip: Annotated[
        bool,
        click.option(
            "--clip",
            type=bool,
            help="Clip vectors to [-65504, 65504] for FAISS FP16",
            default=False,
            required=False,
        ),
    ]

    engine: Annotated[
        str | None,
        click.option(
            "--engine",
            type=click.Choice(["faiss", "lucene"]),
            help="quantization type for vectors (in index)",
            default="faiss",
            required=False,
        ),
    ]

    number_of_indexing_clients: Annotated[
        int,
        click.option(
            "--number-of-indexing-clients",
            type=int,
            help="Number of concurrent clients for data insertion",
            default=1,
        ),
    ]

    use_local_preference: Annotated[
        bool,
        click.option(
            "--use-local-preference",
            type=bool,
            help="Use _only_local search preference for single-shard indices (disable for managed/cloud deployments)",
            default=True,
        ),
    ]


class OSSOpenSearchHNSWTypedDict(CommonTypedDict, OSSOpenSearchTypedDict, HNSWFlavor1): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(OSSOpenSearchHNSWTypedDict)
def OSSOpenSearch(**parameters: Unpack[OSSOpenSearchHNSWTypedDict]):
    from .config import OSSOpenSearchConfig, OSSOpenSearchIndexConfig

    run(
        db=DB.OSSOpenSearch,
        db_config=OSSOpenSearchConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
            use_ssl=parameters.get("use_ssl"),
            db_label=parameters["db_label"],
        ),
        db_case_config=OSSOpenSearchIndexConfig(
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            index_thread_qty=parameters["index_thread_qty"],
            number_of_segments=parameters["number_of_segments"],
            refresh_interval=parameters["refresh_interval"],
            force_merge_enabled=parameters["force_merge_enabled"],
            flush_threshold_size=parameters["flush_threshold_size"],
            index_thread_qty_during_force_merge=parameters["index_thread_qty_during_force_merge"],
            cb_threshold=parameters["cb_threshold"],
            efConstruction=parameters["ef_construction"],
            efSearch=parameters["ef_search"],
            M=parameters["m"],
            engine=OSSOS_Engine(parameters["engine"]),
            quantization_type=OSSOpenSearchQuantization(parameters["quantization_type"]),
            confidence_interval=parameters["confidence_interval"],
            clip=parameters["clip"],
            metric_type_name=parameters["metric_type"],
            number_of_indexing_clients=parameters["number_of_indexing_clients"],
            use_local_preference=parameters["use_local_preference"],
        ),
        **parameters,
    )
