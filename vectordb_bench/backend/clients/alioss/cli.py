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
from .config import AliOSSIndexConfig


class AliOSSTypedDict(TypedDict):
    access_key_id: Annotated[
        str, click.option("--access-key-id", type=str, help="Aliyun AccessKey ID", required=True)
    ]
    access_key_secret: Annotated[
        str, click.option("--access-key-secret", type=str, help="Aliyun AccessKey Secret", required=True)
    ]
    region: Annotated[
        str, click.option("--region", type=str, help="Aliyun region (e.g. cn-shenzhen)", default="cn-shenzhen")
    ]
    account_id: Annotated[
        str, click.option("--account-id", type=str, help="Aliyun account ID (12-digit number)", required=True)
    ]
    bucket: Annotated[
        str, click.option("--bucket", type=str, help="OSS Vector Bucket name", required=True)
    ]
    index: Annotated[
        str, click.option("--index", type=str, help="Vector index name", default="vdbbench-index")
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=click.Choice(["cosine", "l2"]),
            help="Distance metric (cosine or l2)",
            default="cosine",
            show_default=True,
        ),
    ]
    insert_batch_size: Annotated[
        int,
        click.option(
            "--insert-batch-size",
            type=int,
            help="PutVectors batch size; keep <=100 for 768-dim vectors",
            default=100,
            show_default=True,
        ),
    ]


class AliOSSIndexTypedDict(CommonTypedDict, AliOSSTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AliOSSIndexTypedDict)
def AliOSS(**parameters: Unpack[AliOSSIndexTypedDict]):
    from .config import AliOSSConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.AliOSS,
        db_config=AliOSSConfig(
            access_key_id=SecretStr(parameters["access_key_id"]),
            access_key_secret=SecretStr(parameters["access_key_secret"]),
            region=parameters["region"],
            account_id=parameters["account_id"],
            bucket_name=parameters["bucket"],
            index_name=parameters["index"] if parameters["index"] else "vdbbench-index",
            insert_batch_size=parameters["insert_batch_size"],
        ),
        db_case_config=AliOSSIndexConfig(
            metric_type=MetricType.COSINE if parameters["metric"] == "cosine" else MetricType.L2,
        ),
        **parameters,
    )
