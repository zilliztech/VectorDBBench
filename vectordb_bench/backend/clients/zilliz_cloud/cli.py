import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


def _use_partition_key(parameters: dict) -> bool:
    explicit = parameters.get("use_partition_key")
    if explicit is not None:
        return explicit
    return parameters.get("case_type") == "CloudMultiTenantSearchCase"


class ZillizTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", default=""),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Zilliz password",
            default=lambda: os.environ.get("ZILLIZ_PASSWORD", ""),
            show_default="$ZILLIZ_PASSWORD",
        ),
    ]
    token: Annotated[
        str,
        click.option(
            "--token",
            type=str,
            help="Zilliz API token",
            default=lambda: os.environ.get("ZILLIZ_TOKEN", ""),
            show_default="$ZILLIZ_TOKEN",
        ),
    ]
    level: Annotated[
        str,
        click.option("--level", type=str, help="Zilliz index level", required=False),
    ]
    num_shards: Annotated[
        int,
        click.option(
            "--num-shards",
            type=int,
            help="Number of shards",
            required=False,
            default=1,
            show_default=True,
        ),
    ]
    collection_name: Annotated[
        str,
        click.option(
            "--collection-name",
            type=str,
            help="Collection name for Zilliz",
            required=False,
            default="ZillizCloudVDBBench",
            show_default=True,
        ),
    ]
    use_partition_key: Annotated[
        bool | None,
        click.option(
            "--use-partition-key/--no-use-partition-key",
            default=None,
            help=(
                "Use the Zilliz Cloud partition key on the label field. "
                "Defaults to enabled for CloudMultiTenantSearchCase and disabled otherwise."
            ),
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(ZillizTypedDict)
def ZillizAutoIndex(**parameters: Unpack[ZillizTypedDict]):
    from .config import AutoIndexConfig, ZillizCloudConfig

    run(
        db=DB.ZillizCloud,
        db_config=ZillizCloudConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]),
            token=SecretStr(parameters["token"]),
            num_shards=parameters["num_shards"],
            collection_name=parameters["collection_name"],
        ),
        db_case_config=AutoIndexConfig(
            level=int(parameters["level"]) if parameters["level"] else 1,
            num_shards=parameters["num_shards"],
            use_partition_key=_use_partition_key(parameters),
        ),
        **parameters,
    )
