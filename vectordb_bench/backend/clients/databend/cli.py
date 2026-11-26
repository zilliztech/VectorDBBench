from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor2,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from .config import DatabendIndexConfig


class DatabendTypedDict(TypedDict):
    password: Annotated[str, click.option("--password", type=str, help="DB password")]
    host: Annotated[str, click.option("--host", type=str, help="DB host", required=True)]
    port: Annotated[int, click.option("--port", type=int, default=8000, help="DB Port")]
    user: Annotated[str, click.option("--user", type=str, default="root", help="DB user")]
    database: Annotated[
        str, click.option("--database", type=str, default="default", help="Database of the benchmark table")
    ]
    secure: Annotated[bool, click.option("--secure", is_flag=True, default=False, help="Connect over TLS")]
    refine_factor: Annotated[
        int | None,
        click.option(
            "--refine-factor",
            type=int,
            default=None,
            help="Create the index with refine = true (FP16 data) and query with this vector_refine_factor",
        ),
    ]
    create_index_after_load: Annotated[
        bool,
        click.option(
            "--create-index-after-load",
            is_flag=True,
            default=False,
            help="Create the vector index in optimize, after the load, instead of before it",
        ),
    ]
    recluster: Annotated[
        bool,
        click.option(
            "--recluster",
            is_flag=True,
            default=False,
            help="In optimize, CLUSTER BY the vector column and RECLUSTER FINAL (k-means block layout)",
        ),
    ]


class DatabendHNSWTypedDict(CommonTypedDict, DatabendTypedDict, HNSWFlavor2): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(DatabendHNSWTypedDict)
def Databend(**parameters: Unpack[DatabendHNSWTypedDict]):
    from .config import DatabendConfig

    run(
        db=DB.Databend,
        db_config=DatabendConfig(
            db_label=parameters["db_label"],
            user=parameters["user"],
            password=SecretStr(parameters["password"] or ""),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["database"],
            secure=parameters["secure"],
        ),
        db_case_config=DatabendIndexConfig(
            metric_type=None,
            m=parameters["m"],
            ef_construct=parameters["ef_construction"],
            refine_factor=parameters["refine_factor"],
            ef_search=parameters["ef_runtime"],
            create_index_before_load=not parameters["create_index_after_load"],
            create_index_after_load=parameters["create_index_after_load"],
            recluster=parameters["recluster"],
        ),
        **parameters,
    )
