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


class DynamoDBTypedDict(TypedDict):
    region_name: Annotated[
        str,
        click.option("--region", type=str, help="AWS region (e.g. us-east-1)", default="us-east-1"),
    ]
    access_key_id: Annotated[
        str,
        click.option(
            "--access_key_id",
            type=str,
            help="AWS access key ID. Omit to use the default boto3 credential chain.",
            default=None,
        ),
    ]
    secret_access_key: Annotated[
        str,
        click.option(
            "--secret_access_key",
            type=str,
            help="AWS secret access key. Omit to use the default boto3 credential chain.",
            default=None,
        ),
    ]
    session_token: Annotated[
        str,
        click.option(
            "--session_token",
            type=str,
            help="Optional AWS session token for temporary credentials.",
            default=None,
        ),
    ]
    table: Annotated[
        str,
        click.option("--table", type=str, help="DynamoDB table name", default="vdbbench_vectors"),
    ]
    index: Annotated[
        str,
        click.option("--index", type=str, help="Vector index name", default="vdbbench-index"),
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=str,
            help="Distance metric: cosine, euclidean (l2), or dotproduct (ip).",
            default="cosine",
        ),
    ]
    partition_count: Annotated[
        int,
        click.option(
            "--partition-count",
            type=int,
            help=(
                "Number of vector-index partition-key (SearchSchema HASH) values. "
                ">1 defines a partition key; each search is scoped to one randomly "
                "chosen partition value (realistic single-partition access, recall "
                "~1/N vs whole-dataset ground truth). 1 (default) searches the whole "
                "index with no partition key."
            ),
            default=1,
        ),
    ]


class DynamoDBIndexTypedDict(CommonTypedDict, DynamoDBTypedDict): ...


_METRIC_MAP = {
    "cosine": MetricType.COSINE,
    "euclidean": MetricType.L2,
    "l2": MetricType.L2,
    "dotproduct": MetricType.IP,
    "ip": MetricType.IP,
}


@cli.command()
@click_parameter_decorators_from_typed_dict(DynamoDBIndexTypedDict)
def DynamoDB(**parameters: Unpack[DynamoDBIndexTypedDict]):
    from .config import DynamoDBConfig, DynamoDBIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.DynamoDB,
        db_config=DynamoDBConfig(
            region_name=parameters["region"],
            access_key_id=(SecretStr(parameters["access_key_id"]) if parameters["access_key_id"] else None),
            secret_access_key=(
                SecretStr(parameters["secret_access_key"]) if parameters["secret_access_key"] else None
            ),
            session_token=(SecretStr(parameters["session_token"]) if parameters["session_token"] else None),
            table_name=parameters["table"],
            index_name=parameters["index"] if parameters["index"] else "vdbbench-index",
        ),
        db_case_config=DynamoDBIndexConfig(
            metric_type=_METRIC_MAP.get((parameters["metric"] or "cosine").lower()),
            num_partitions=parameters.get("partition_count", 1) or 1,
        ),
        **parameters,
    )
