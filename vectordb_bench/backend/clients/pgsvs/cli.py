import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr, ValidationError

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType

from ....cli.cli import (
    CommonTypedDict,
    SVSBaseTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class PgSVSTypedDict(CommonTypedDict, SVSBaseTypedDict):
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
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
            required=False,
        ),
    ]
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
    quantization_type: Annotated[
        str | None,
        click.option(
            "--quantization-type",
            type=click.Choice(["none", "bit", "halfvec"]),
            help="quantization type for vectors (in index)",
            required=False,
        ),
    ]
    table_quantization_type: Annotated[
        str | None,
        click.option(
            "--table-quantization-type",
            type=click.Choice(["none", "bit", "halfvec"]),
            help="quantization type for vectors (in table). "
            "If equal to bit, the parameter quantization_type will be set to bit too.",
            required=False,
        ),
    ]
    reranking: Annotated[
        bool | None,
        click.option(
            "--reranking/--skip-reranking",
            type=bool,
            help="Enable reranking for HNSW search for binary quantization",
            default=False,
        ),
    ]
    reranking_metric: Annotated[
        str | None,
        click.option(
            "--reranking-metric",
            type=click.Choice(
                [metric.value for metric in MetricType if metric.value not in ["HAMMING", "JACCARD"]],
            ),
            help="Distance metric for reranking",
            default="COSINE",
            show_default=True,
        ),
    ]
    quantized_fetch_limit: Annotated[
        int | None,
        click.option(
            "--quantized-fetch-limit",
            type=int,
            help="Limit of fetching quantized vector ranked by distance for reranking \
                -- bound by ef_search",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(PgSVSTypedDict)
def PgSVS(
    **parameters: Unpack[PgSVSTypedDict],
):
    from .config import PgSVSConfig, PgSVSVamanaConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    try:
        case_config = PgSVSVamanaConfig(
            graph_degree=parameters["graph_degree"],
            alpha=parameters["alpha"],
            search_window_size=parameters["search_window_size"],
            use_search_history=parameters["use_search_history"],
            build_window_size=parameters["build_window_size"],
            leanvec_dims=parameters["leanvec_dims"],
            compression_type=parameters["compression_type"],
            compression_primary=parameters["compression_primary"],
            compression_secondary=parameters["compression_secondary"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            normalize_cosine=parameters["normalize_cosine"],
            quantization_type=parameters["quantization_type"],
            table_quantization_type=parameters["table_quantization_type"],
            reranking=parameters["reranking"],
            quantized_fetch_limit=parameters["quantized_fetch_limit"],
        )
    except ValidationError as e:
        raise click.UsageError(str(e)) from e
    run(
        db=DB.PgSVS,
        db_config=PgSVSConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=case_config,
        **parameters,
    )
