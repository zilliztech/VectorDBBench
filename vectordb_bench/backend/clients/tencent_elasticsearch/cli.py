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


class TencentElasticsearchTypedDict(CommonTypedDict):
    scheme: Annotated[
        str,
        click.option(
            "--scheme",
            type=str,
            help="Protocol in use to connect to the node",
            default="http",
            show_default=True,
        ),
    ]
    host: Annotated[
        str,
        click.option("--host", type=str, help="shot connection string", required=True),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="Port to connect to", default=9200, show_default=True),
    ]
    user: Annotated[
        str,
        click.option("--user", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="TencentElasticsearch password",
            default=lambda: os.environ.get("TES_PASSWORD", ""),
            show_default="$TES_PASSWORD",
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, help="HNSW M parameter", default=16, show_default=True),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef_construction",
            type=int,
            help="HNSW efConstruction parameter",
            default=200,
            show_default=True,
        ),
    ]
    num_candidates: Annotated[
        int,
        click.option(
            "--num_candidates",
            type=int,
            help="Number of candidates to consider during searching",
            default=200,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TencentElasticsearchTypedDict)
def TencentElasticsearch(**parameters: Unpack[TencentElasticsearchTypedDict]):
    from .config import TencentElasticsearchConfig, TencentElasticsearchIndexConfig

    run(
        db=DB.TencentElasticsearch,
        db_config=TencentElasticsearchConfig(
            db_label=parameters["db_label"],
            scheme=parameters["scheme"],
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=TencentElasticsearchIndexConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            num_candidates=parameters["num_candidates"],
        ),
        **parameters,
    )
