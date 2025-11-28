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
from ..api import MetricType
from .config import TurbopufferIndexConfig


class TurbopufferTypedDict(TypedDict):
    api_key: Annotated[
        str, click.option("--api-key", type=str, help="Turbopuffer API key", required=True)
    ]
    region: Annotated[
        str,
        click.option(
            "--region",
            type=str,
            help="Turbopuffer region (e.g., aws-us-east-1)",
            default="aws-us-east-1",
            show_default=True
        )
    ]
    namespace: Annotated[
        str,
        click.option(
            "--namespace",
            type=str,
            help="Turbopuffer namespace",
            default="vdbbench",
            show_default=True
        )
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=str,
            help="Distance metric for vector similarity (cosine, l2, ip)",
            default=None,
        ),
    ]


class TurbopufferIndexTypedDict(CommonTypedDict, TurbopufferTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(TurbopufferIndexTypedDict)
def Turbopuffer(**parameters: Unpack[TurbopufferIndexTypedDict]):
    from .config import TurbopufferConfig

    metric_type = None
    if parameters.get("metric"):
        metric = parameters["metric"].lower()
        if metric == "cosine":
            metric_type = MetricType.COSINE
        elif metric == "l2":
            metric_type = MetricType.L2
        elif metric == "ip":
            metric_type = MetricType.IP

    run(
        db=DB.Turbopuffer,
        db_config=TurbopufferConfig(
            api_key=SecretStr(parameters["api_key"]),
            region=parameters["region"],
            namespace=parameters["namespace"],
        ),
        db_case_config=TurbopufferIndexConfig(
            metric_type=metric_type
        ),
        **parameters,
    )