"""HouyiDB command-line entry point for VectorDBBench."""

from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..api import MetricType


class HouyiDBTypedDict(CommonTypedDict):
    host: Annotated[
        str,
        click.option("--host", type=str, required=True, help="HouyiDB host"),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=15432,
            show_default=True,
            help="HouyiDB pgwire port",
        ),
    ]
    http_port: Annotated[
        int | None,
        click.option(
            "--http-port",
            type=int,
            default=17000,
            show_default=True,
            help=("Benchmark HTTP port for health, optimize, index status and the HTTP data plane"),
        ),
    ]
    transport: Annotated[
        str,
        click.option(
            "--transport",
            type=click.Choice(["pgwire", "http"], case_sensitive=False),
            default="pgwire",
            show_default=True,
            help="Data plane: pgwire or the HTTP benchmark API",
        ),
    ]
    use_tls: Annotated[
        bool,
        click.option(
            "--use-tls/--no-use-tls",
            default=False,
            show_default=True,
            help="Use HTTPS for HTTP requests",
        ),
    ]
    api_key: Annotated[
        str,
        click.option(
            "--api-key",
            envvar="HOUYI_API_KEY",
            default="",
            show_default=False,
            help="Optional token: HTTP Bearer or pgwire password",
        ),
    ]
    timeout_sec: Annotated[
        int,
        click.option(
            "--timeout-sec",
            type=int,
            default=600,
            show_default=True,
            help="Per-request timeout in seconds",
        ),
    ]
    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice(
                [metric.name for metric in (MetricType.L2, MetricType.COSINE, MetricType.IP)],
                case_sensitive=True,
            ),
            default=MetricType.COSINE.name,
            show_default=True,
            help="Distance metric",
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, default=24, show_default=True, help="HNSW m"),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            default=400,
            show_default=True,
            help="HNSW ef_construction",
        ),
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=int,
            default=256,
            show_default=True,
            help="HNSW ef_search",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(HouyiDBTypedDict)
def HouyiDB(**parameters: Unpack[HouyiDBTypedDict]):
    """Run a VectorDBBench case against HouyiDB."""
    from .config import HouyiDBConfig, HouyiDBIndexConfig

    run(
        db=DB.HouyiDB,
        db_config=HouyiDBConfig(
            db_label=parameters["db_label"],
            note=parameters["note"],
            host=parameters["host"],
            port=parameters["port"],
            http_port=parameters["http_port"],
            use_tls=parameters["use_tls"],
            transport=parameters["transport"].lower(),
            api_key=SecretStr(parameters["api_key"]) if parameters["api_key"] else None,
            timeout_sec=parameters["timeout_sec"],
        ),
        db_case_config=HouyiDBIndexConfig(
            metric_type=MetricType[parameters["metric_type"]],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
