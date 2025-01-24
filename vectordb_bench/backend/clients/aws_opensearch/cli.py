from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor2,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class AWSOpenSearchTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=443, help="Db Port")]
    user: Annotated[str, click.option("--user", type=str, default="admin", help="Db User")]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    number_of_shards: Annotated[
        int,
        click.option("--number-of-shards", type=int, help="Number of shards", default=1),
    ]
    number_of_replicas: Annotated[
        int,
        click.option("--number-of-replicas", type=int, help="Number of replica", default=1),
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
            "--index-thread-qty-during-force-merge",
            type=int,
            help="Thread count for native engine indexing used during force merge",
            default=4,
        ),
    ]

    number_of_segments: Annotated[
        int,
        click.option("--number-of-segments", type=int, help="Number of segments", default=1),
    ]

    refresh_interval: Annotated[
        int,
        click.option("--refresh-interval", type=str, help="refresh-interval", default="60s"),
    ]

    force_merge_enabled: Annotated[
        int,
        click.option("--force-merge-enabled", type=bool, help="If we need to do force merge or not", default=True),
    ]

    flush_threshold_size: Annotated[
        int,
        click.option("--flush-threshold-size", type=str, help="Threshold for flushing translog", default="5120mb"),
    ]

    number_of_indexing_clients: Annotated[
        int,
        click.option(
            "--number-of-indexing-clients",
            type=int,
            help="Number of indexing clients that should be used for indexing the data",
            default=1,
        ),
    ]

    cb_threshold: Annotated[
        int,
        click.option(
            "--cb-threshold",
            type=str,
            help="k-NN Memory circuit breaker threshold",
            default="50%",
        ),
    ]


class AWSOpenSearchHNSWTypedDict(CommonTypedDict, AWSOpenSearchTypedDict, HNSWFlavor2): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AWSOpenSearchHNSWTypedDict)
def AWSOpenSearch(**parameters: Unpack[AWSOpenSearchHNSWTypedDict]):
    from .config import AWSOpenSearchConfig, AWSOpenSearchIndexConfig

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
            number_of_indexing_clients=parameters["number_of_indexing_clients"],
            index_thread_qty_during_force_merge=parameters["index_thread_qty_during_force_merge"],
            cb_threshold=parameters["cb_threshold"],
        ),
        **parameters,
    )
