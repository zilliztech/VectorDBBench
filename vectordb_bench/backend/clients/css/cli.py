import logging
from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from .config import CSS_Engine

log = logging.getLogger(__name__)


class CSSTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=80, help="Db Port")]
    user: Annotated[str, click.option("--user", type=str, help="Db User")]
    password: Annotated[str | None, click.option("--password", type=str, help="Db password", default=None)]
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
            "--index-thread-qty-during-force-merge",
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
            default="cosine",
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

    engine: Annotated[
        str | None,
        click.option(
            "--engine",
            type=click.Choice(["hanns"]),
            help="engine type for vectors (in index)",
            default="hanns",
            required=False,
        ),
    ]
    # HANNS build parameters
    max_degree: Annotated[
        int | None,
        click.option("--max-degree", type=int, help="HANNS max_degree (2-512)", default=56),
    ]
    search_list_size_build: Annotated[
        int | None,
        click.option("--search-list-size-build", type=int, help="HANNS build search_list_size (1-10000)", default=200),
    ]
    encoder: Annotated[
        str,
        click.option(
            "--encoder",
            type=click.Choice(["sq8", "extended-rabitq"]),
            help="HANNS encoder",
            default="sq8",
        ),
    ]
    nbit: Annotated[
        str,
        click.option(
            "--nbit",
            type=click.Choice(["1", "2", "4", "8"]),
            help="HANNS encoder nbit (for extended-rabitq)",
            default="4",
        ),
    ]
    pca_dim: Annotated[
        int,
        click.option(
            "--pca-dim",
            type=int,
            help="PCA dimension for HANNS (0=disabled)",
            default=0,
            show_default=True,
        ),
    ]
    # HANNS search parameters
    search_list_size: Annotated[
        int,
        click.option("--search-list-size", type=int, help="HANNS search search_list_size (1-10000)", default=100),
    ]

class CSSHANNSTypedDict(CommonTypedDict, CSSTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(CSSHANNSTypedDict)
def CSS(**parameters: Unpack[CSSHANNSTypedDict]):
    from .config import CSSConfig, CSSIndexConfig

    encoder_config = {"name": parameters["encoder"]}
    if parameters["encoder"] == "extended-rabitq":
        encoder_config["parameters"] = {"nbit": int(parameters["nbit"])}

    # Build case_config kwargs, omitting None values so pydantic uses field defaults
    case_config_kwargs = {
        "number_of_shards": parameters["number_of_shards"],
        "number_of_replicas": parameters["number_of_replicas"],
        "index_thread_qty": parameters["index_thread_qty"],
        "number_of_segments": parameters["number_of_segments"],
        "refresh_interval": parameters["refresh_interval"],
        "force_merge_enabled": parameters["force_merge_enabled"],
        "flush_threshold_size": parameters["flush_threshold_size"],
        "index_thread_qty_during_force_merge": parameters["index_thread_qty_during_force_merge"],
        "cb_threshold": parameters["cb_threshold"],
        "engine": CSS_Engine(parameters["engine"]),
        "metric_type_name": parameters["metric_type"],
        # HANNS parameters
        "max_degree": parameters["max_degree"],
        "search_list_size_build": parameters["search_list_size_build"],
        "search_list_size": parameters["search_list_size"],
        "encoder": encoder_config,
        "pca_dim": parameters["pca_dim"],
    }

    run(
        db=DB.CSS,
        db_config=CSSConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]) if parameters.get("password") else None,
        ),
        db_case_config=CSSIndexConfig(**case_config_kwargs),
        **parameters,
    )
