from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .. import DB
from ..api import MetricType
from .config import S3VectorsIndexConfig


class S3VectorsTypedDict(TypedDict):
    region_name: Annotated[
        str, click.option("--region", type=str, help="AWS region for S3 bucket (eg. us-east-1)", default="us-east-1")
    ]
    access_key_id: Annotated[str, click.option("--access_key_id", type=str, help="AWS access key ID", required=True)]
    secret_access_key: Annotated[
        str, click.option("--secret_access_key", type=str, help="AWS secret access key", required=True)
    ]

    bucket: Annotated[str, click.option("--bucket", type=str, help="S3 bucket name", required=True)]
    index: Annotated[str, click.option("--index", type=str, help="Unique vector index name", default="vdbbench-index")]

    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=str,
            help="Distance metric for vector similarity (e.g., 'cosine', 'euclidean').",
            default=None,
        ),
    ]


class S3VectorsIndexTypedDict(CommonTypedDict, S3VectorsTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(S3VectorsIndexTypedDict)
def S3Vectors(**parameters: Unpack[S3VectorsIndexTypedDict]):
    from .config import S3VectorsConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.S3Vectors,
        db_config=S3VectorsConfig(
            region_name=parameters["region"],
            access_key_id=SecretStr(parameters["access_key_id"]),
            secret_access_key=SecretStr(parameters["secret_access_key"]),
            bucket_name=parameters["bucket"],
            index_name=parameters["index"] if parameters["index"] else "vdbbench-index",
        ),
        db_case_config=S3VectorsIndexConfig(
            metric_type=(
                MetricType.COSINE
                if parameters["metric"] == "cosine"
                else MetricType.L2 if parameters["metric"] == "l2" else None
            )
        ),
        **parameters,
    )
