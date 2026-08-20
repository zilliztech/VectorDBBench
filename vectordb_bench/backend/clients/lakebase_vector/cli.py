import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run


class LakebaseVectorTypedDict(CommonTypedDict):
    user_name: Annotated[str, click.option("--user-name", type=str, required=True)]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]
    host: Annotated[str, click.option("--host", type=str, required=True)]
    port: Annotated[int, click.option("--port", type=int, default=5432, show_default=True)]
    db_name: Annotated[str, click.option("--db-name", type=str, default="databricks_postgres", show_default=True)]
    table_name: Annotated[
        str,
        click.option("--table-name", type=str, default="vdbbench_table_test", show_default=True),
    ]
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Set max_parallel_maintenance_workers and max_parallel_workers for index creation",
        ),
    ]
    probes: Annotated[
        str | None,
        click.option(
            "--probes",
            type=str,
            help="Comma-separated lakebase_ann probe counts; omit to use index defaults",
        ),
    ]
    epsilon: Annotated[
        float | None,
        click.option(
            "--epsilon",
            type=float,
            default=None,
            help="Lakebase ANN reranking margin; omit to use the index default",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LakebaseVectorTypedDict)
def LakebaseANN(**parameters: Unpack[LakebaseVectorTypedDict]):
    from .config import LakebaseANNConfig, LakebaseVectorConfig

    run(
        db=DB.LakebaseVector,
        db_config=LakebaseVectorConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            table_name=parameters["table_name"],
        ),
        db_case_config=LakebaseANNConfig(
            metric_type=None,
            probes=parameters["probes"],
            epsilon=parameters["epsilon"],
            max_parallel_workers=parameters["max_parallel_workers"],
        ),
        **parameters,
    )
