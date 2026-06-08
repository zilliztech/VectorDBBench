from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

from ..milvus.cli import MilvusDISKANNTypedDict, _with_partition_key

DBTYPE = DB.AliyunMilvus


class AliyunMilvusDISKANNTypedDict(MilvusDISKANNTypedDict):
    """Same as Milvus DISKANN, plus three opt-in search-time params.

    Each of the three options defaults to "not specified" (omit -> not sent).
    """

    rerank_topk_multiplier: Annotated[
        int | None,
        click.option(
            "--rerank-topk-multiplier",
            type=int,
            help="Search param: topk multiplier for rerank budget (0 disables rerank read). Omit to not send it.",
            required=False,
            default=None,
        ),
    ]
    early_termination_threshold: Annotated[
        int | None,
        click.option(
            "--early-termination-threshold",
            type=int,
            help="Search param: early termination threshold (0 disables). Omit to not send it.",
            required=False,
            default=None,
        ),
    ]
    cross_segment_rerank: Annotated[
        bool | None,
        click.option(
            "--cross-segment-rerank/--no-cross-segment-rerank",
            "cross_segment_rerank",
            help="Search param: enable cross-segment rerank. Omit to not send it.",
            default=None,
        ),
    ]


@cli.command(name="aliyunmilvusdiskann")
@click_parameter_decorators_from_typed_dict(AliyunMilvusDISKANNTypedDict)
def AliyunMilvusDISKANN(**parameters: Unpack[AliyunMilvusDISKANNTypedDict]):
    from ..milvus.config import MilvusConfig
    from .config import AliyunMilvusDISKANNConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=int(parameters["num_shards"]),
            replica_number=int(parameters["replica_number"]),
        ),
        db_case_config=_with_partition_key(
            AliyunMilvusDISKANNConfig(
                search_list=parameters["search_list"],
                # Pass through as-is; None means "not specified" -> omitted from search params.
                rerank_topk_multiplier=parameters["rerank_topk_multiplier"],
                early_termination_threshold=parameters["early_termination_threshold"],
                cross_segment_rerank=parameters["cross_segment_rerank"],
            ),
            parameters,
        ),
        **parameters,
    )
