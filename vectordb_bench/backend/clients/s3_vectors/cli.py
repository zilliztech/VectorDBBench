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

    insert_batch_size: Annotated[
        int,
        click.option(
            "--insert-batch-size",
            type=int,
            help="PutVectors batch size; AWS hard limit 500 per call",
            default=100,
            show_default=True,
        ),
    ]
    max_pool_connections: Annotated[
        int,
        click.option(
            "--max-pool-connections",
            type=int,
            help="urllib3 connection pool size; should be >= 2x ConcurrentInsertRunner worker count",
            default=50,
            show_default=True,
        ),
    ]
    retry_mode: Annotated[
        str,
        click.option(
            "--retry-mode",
            type=click.Choice(["legacy", "standard", "adaptive"]),
            help="boto3 retry mode; adaptive recommended for throttling resilience",
            default="adaptive",
            show_default=True,
        ),
    ]
    retry_max_attempts: Annotated[
        int,
        click.option(
            "--retry-max-attempts",
            type=int,
            help="boto3 retry total attempt count (including the first call)",
            default=10,
            show_default=True,
        ),
    ]
    endpoint_url: Annotated[
        str | None,
        click.option(
            "--endpoint-url",
            type=str,
            help="Custom S3 Vectors endpoint URL (e.g. http://192.168.1.100:8080)",
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
            endpoint_url=parameters.get("endpoint_url"),
            insert_batch_size=parameters["insert_batch_size"],
            max_pool_connections=parameters["max_pool_connections"],
            retry_mode=parameters["retry_mode"],
            retry_max_attempts=parameters["retry_max_attempts"],
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
