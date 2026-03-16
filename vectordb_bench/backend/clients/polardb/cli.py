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


class PolarDBTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            required=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Password",
            default="",
        ),
    ]

    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="Db host",
            default="127.0.0.1",
        ),
    ]

    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=3306,
            help="Db Port",
        ),
    ]

    database: Annotated[
        str,
        click.option(
            "--database",
            type=str,
            help="Database name",
            default="vectordbbench",
        ),
    ]

    unix_socket: Annotated[
        str,
        click.option(
            "--unix-socket",
            type=str,
            help="Unix socket path (overrides host/port if set)",
            default="",
        ),
    ]


class PolarDBHNSWTypedDict(PolarDBTypedDict):
    m: Annotated[
        int,
        click.option(
            "--m",
            type=int,
            help="M parameter (max_degree) in HNSW",
            default=16,
        ),
    ]

    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="ef_construction parameter in HNSW",
            default=200,
        ),
    ]

    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=int,
            help="polar_vector_index_hnsw_ef_search session variable",
            default=64,
        ),
    ]

    insert_workers: Annotated[
        int,
        click.option(
            "--insert-workers",
            type=int,
            help="Number of concurrent threads for data insertion",
            default=10,
        ),
    ]

    post_load_index: Annotated[
        bool,
        click.option(
            "--post-load-index/--inline-index",
            type=bool,
            help="If set, create vector index via ALTER TABLE after data load; otherwise create index inline during table creation",
            default=False,
        ),
    ]


class PolarDBHNSWPQTypedDict(PolarDBHNSWTypedDict):
    pq_m: Annotated[
        int,
        click.option(
            "--pq-m",
            type=int,
            help="PQ subquantizer count (must divide dimension)",
            default=1,
        ),
    ]

    pq_nbits: Annotated[
        int,
        click.option(
            "--pq-nbits",
            type=int,
            help="PQ bits per subquantizer (max 24)",
            default=8,
        ),
    ]


class PolarDBHNSWSQTypedDict(PolarDBHNSWTypedDict):
    sq_type: Annotated[
        str,
        click.option(
            "--sq-type",
            type=str,
            help="SQ quantizer type (8bit, 4bit, fp16, bf16, 6bit, etc.)",
            default="8bit",
        ),
    ]


def _build_db_config(parameters):
    from .config import PolarDBConfig

    pwd = parameters["password"]
    sock = parameters["unix_socket"]
    return PolarDBConfig(
        db_label=parameters["db_label"],
        user_name=parameters["username"],
        password=SecretStr(pwd) if pwd else None,
        host=parameters["host"],
        port=parameters["port"],
        database=parameters["database"],
        unix_socket=sock if sock else None,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(PolarDBHNSWTypedDict)
def PolarDBHNSWFlat(
    **parameters: Unpack[PolarDBHNSWTypedDict],
):
    from .config import PolarDBHNSWFlatConfig

    run(
        db=DB.PolarDB,
        db_config=_build_db_config(parameters),
        db_case_config=PolarDBHNSWFlatConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            insert_workers=parameters["insert_workers"],
            post_load_index=parameters["post_load_index"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(PolarDBHNSWPQTypedDict)
def PolarDBHNSWPQ(
    **parameters: Unpack[PolarDBHNSWPQTypedDict],
):
    from .config import PolarDBHNSWPQConfig

    run(
        db=DB.PolarDB,
        db_config=_build_db_config(parameters),
        db_case_config=PolarDBHNSWPQConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            insert_workers=parameters["insert_workers"],
            post_load_index=parameters["post_load_index"],
            pq_m=parameters["pq_m"],
            pq_nbits=parameters["pq_nbits"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(PolarDBHNSWSQTypedDict)
def PolarDBHNSWSQ(
    **parameters: Unpack[PolarDBHNSWSQTypedDict],
):
    from .config import PolarDBHNSWSQConfig

    run(
        db=DB.PolarDB,
        db_config=_build_db_config(parameters),
        db_case_config=PolarDBHNSWSQConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            insert_workers=parameters["insert_workers"],
            post_load_index=parameters["post_load_index"],
            sq_type=parameters["sq_type"],
        ),
        **parameters,
    )
