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


class TurboPufferTypedDict(TypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="TurboPuffer API key", required=True),
    ]
    region: Annotated[
        str,
        click.option(
            "--region",
            type=str,
            help="TurboPuffer region (e.g. aws-us-east-1, gcp-us-central1)",
            required=True,
        ),
    ]
    api_base_url: Annotated[
        str,
        click.option(
            "--api-base-url",
            type=str,
            help="Override the region-based API URL",
            required=False,
            default="",
            show_default=False,
        ),
    ]
    namespace: Annotated[
        str,
        click.option(
            "--namespace",
            type=str,
            help="TurboPuffer namespace",
            required=False,
            default="vdbbench_test",
            show_default=True,
        ),
    ]
    multitenant_namespace_prefix: Annotated[
        str,
        click.option(
            "--multitenant-namespace-prefix",
            type=str,
            help="Namespace prefix for CloudMultiTenantSearchCase tenant namespaces",
            required=False,
            default="vdbbench_mt_",
            show_default=True,
        ),
    ]
    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice([MetricType.COSINE.value, MetricType.L2.value]),
            help="TurboPuffer distance metric type",
            required=False,
            default=MetricType.COSINE.value,
            show_default=True,
        ),
    ]
    disable_backpressure: Annotated[
        bool,
        click.option(
            "--disable-backpressure/--enable-backpressure",
            type=bool,
            default=False,
            show_default=True,
            help="Disable Turbopuffer write backpressure",
        ),
    ]


class TurboPufferIndexTypedDict(CommonTypedDict, TurboPufferTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(TurboPufferIndexTypedDict)
def TurboPuffer(**parameters: Unpack[TurboPufferIndexTypedDict]):
    from .config import TurboPufferConfig, TurboPufferIndexConfig

    run(
        db=DB.TurboPuffer,
        db_config=TurboPufferConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]),
            region=parameters["region"],
            api_base_url=parameters["api_base_url"] or None,
            namespace=parameters["namespace"],
            multitenant_namespace_prefix=parameters["multitenant_namespace_prefix"],
        ),
        db_case_config=TurboPufferIndexConfig(
            metric_type=MetricType(parameters["metric_type"]),
            disable_backpressure=parameters["disable_backpressure"],
        ),
        **parameters,
    )
