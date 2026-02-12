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
from .config import AstraDBIndexConfig


class AstraDBTypedDict(TypedDict):
    api_endpoint: Annotated[
        str,
        click.option(
            "--api-endpoint",
            type=str,
            help="AstraDB API endpoint (e.g., https://<database-id>-<region>.apps.astra.datastax.com)",
            required=True,
        ),
    ]
    token: Annotated[
        str,
        click.option(
            "--token",
            type=str,
            help="AstraDB authentication token",
            required=True,
        ),
    ]
    namespace: Annotated[
        str,
        click.option(
            "--namespace",
            type=str,
            help="AstraDB namespace (keyspace)",
            default="default_keyspace",
            show_default=True,
        ),
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=click.Choice(["cosine", "euclidean", "dot_product"], case_sensitive=False),
            help="Distance metric for vector similarity",
            default="cosine",
            show_default=True,
        ),
    ]


class AstraDBIndexTypedDict(CommonTypedDict, AstraDBTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AstraDBIndexTypedDict)
def AstraDB(**parameters: Unpack[AstraDBIndexTypedDict]):
    from .config import AstraDBConfig

    # Convert metric string to MetricType enum
    metric_map = {
        "cosine": MetricType.COSINE,
        "euclidean": MetricType.L2,
        "dot_product": MetricType.IP,
    }
    metric_type = metric_map.get(parameters["metric"].lower(), MetricType.COSINE)

    run(
        db=DB.AstraDB,
        db_config=AstraDBConfig(
            db_label=parameters["db_label"],
            api_endpoint=parameters["api_endpoint"],
            token=SecretStr(parameters["token"]),
            namespace=parameters["namespace"],
        ),
        db_case_config=AstraDBIndexConfig(
            metric_type=metric_type,
        ),
        **parameters,
    )
