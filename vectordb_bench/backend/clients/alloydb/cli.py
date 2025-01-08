import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class AlloyDBTypedDict(CommonTypedDict):
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
    maintenance_work_mem: Annotated[
        str | None,
        click.option(
            "--maintenance-work-mem",
            type=str,
            help="Sets the maximum memory to be used for maintenance operations (index creation). "
            "Can be entered as string with unit like '64GB' or as an integer number of KB."
            "This will set the parameters: max_parallel_maintenance_workers,"
            " max_parallel_workers & table(parallel_workers)",
            required=False,
        ),
    ]
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel processes per maintenance operation (index creation)",
            required=False,
        ),
    ]


class AlloyDBScaNNTypedDict(AlloyDBTypedDict):
    num_leaves: Annotated[
        int,
        click.option("--num-leaves", type=int, help="Number of leaves", required=True),
    ]
    num_leaves_to_search: Annotated[
        int,
        click.option(
            "--num-leaves-to-search",
            type=int,
            help="Number of leaves to search",
            required=True,
        ),
    ]
    pre_reordering_num_neighbors: Annotated[
        int,
        click.option(
            "--pre-reordering-num-neighbors",
            type=int,
            help="Pre-reordering number of neighbors",
            default=200,
        ),
    ]
    max_top_neighbors_buffer_size: Annotated[
        int,
        click.option(
            "--max-top-neighbors-buffer-size",
            type=int,
            help="Maximum top neighbors buffer size",
            default=20_000,
        ),
    ]
    num_search_threads: Annotated[
        int,
        click.option("--num-search-threads", type=int, help="Number of search threads", default=2),
    ]
    max_num_prefetch_datasets: Annotated[
        int,
        click.option(
            "--max-num-prefetch-datasets",
            type=int,
            help="Maximum number of prefetch datasets",
            default=100,
        ),
    ]
    quantizer: Annotated[
        str,
        click.option(
            "--quantizer",
            type=click.Choice(["SQ8", "FLAT"]),
            help="Quantizer type",
            default="SQ8",
        ),
    ]
    enable_pca: Annotated[
        bool,
        click.option(
            "--enable-pca",
            type=click.Choice(["on", "off"]),
            help="Enable PCA",
            default="on",
        ),
    ]
    max_num_levels: Annotated[
        int,
        click.option(
            "--max-num-levels",
            type=click.Choice(["1", "2"]),
            help="Maximum number of levels",
            default=1,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AlloyDBScaNNTypedDict)
def AlloyDBScaNN(
    **parameters: Unpack[AlloyDBScaNNTypedDict],
):
    from .config import AlloyDBConfig, AlloyDBScaNNConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.AlloyDB,
        db_config=AlloyDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            db_name=parameters["db_name"],
        ),
        db_case_config=AlloyDBScaNNConfig(
            num_leaves=parameters["num_leaves"],
            quantizer=parameters["quantizer"],
            enable_pca=parameters["enable_pca"],
            max_num_levels=parameters["max_num_levels"],
            num_leaves_to_search=parameters["num_leaves_to_search"],
            max_top_neighbors_buffer_size=parameters["max_top_neighbors_buffer_size"],
            pre_reordering_num_neighbors=parameters["pre_reordering_num_neighbors"],
            num_search_threads=parameters["num_search_threads"],
            max_num_prefetch_datasets=parameters["max_num_prefetch_datasets"],
            max_parallel_workers=parameters["max_parallel_workers"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
        ),
        **parameters,
    )
