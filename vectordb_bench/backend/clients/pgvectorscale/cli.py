import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class PgVectorScaleTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Postgres database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]

    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]


class PgVectorScaleDiskAnnTypedDict(PgVectorScaleTypedDict):
    storage_layout: Annotated[
        str,
        click.option(
            "--storage-layout",
            type=str,
            help="Streaming DiskANN storage layout",
        ),
    ]
    num_neighbors: Annotated[
        int,
        click.option(
            "--num-neighbors",
            type=int,
            help="Streaming DiskANN num neighbors",
        ),
    ]
    search_list_size: Annotated[
        int,
        click.option(
            "--search-list-size",
            type=int,
            help="Streaming DiskANN search list size",
        ),
    ]
    max_alpha: Annotated[
        float,
        click.option(
            "--max-alpha",
            type=float,
            help="Streaming DiskANN max alpha",
        ),
    ]
    num_dimensions: Annotated[
        int,
        click.option(
            "--num-dimensions",
            type=int,
            help="Streaming DiskANN num dimensions",
        ),
    ]
    query_search_list_size: Annotated[
        int,
        click.option(
            "--query-search-list-size",
            type=int,
            help="Streaming DiskANN query search list size",
        ),
    ]
    query_rescore: Annotated[
        int,
        click.option(
            "--query-rescore",
            type=int,
            help="Streaming DiskANN query rescore",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectorScaleDiskAnnTypedDict)
def PgVectorScaleDiskAnn(
    **parameters: Unpack[PgVectorScaleDiskAnnTypedDict],
):
    from .config import PgVectorScaleConfig, PgVectorScaleStreamingDiskANNConfig

    run(
        db=DB.PgVectorScale,
        db_config=PgVectorScaleConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgVectorScaleStreamingDiskANNConfig(
            storage_layout=parameters["storage_layout"],
            num_neighbors=parameters["num_neighbors"],
            search_list_size=parameters["search_list_size"],
            max_alpha=parameters["max_alpha"],
            num_dimensions=parameters["num_dimensions"],
            query_search_list_size=parameters["query_search_list_size"],
            query_rescore=parameters["query_rescore"],
        ),
        **parameters,
    )
