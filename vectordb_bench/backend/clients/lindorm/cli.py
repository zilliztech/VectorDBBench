from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run, HNSWFlavor3, IVFFlatTypedDictN,
)
from vectordb_bench.backend.clients import DB


class LindormTypedDict(CommonTypedDict):
    host: Annotated[
        str, click.option("--host", type=str, help="host connection string", required=True)
    ]

    port: Annotated[int, click.option("--port", type=int, default=30070, help="Db Port")]

    user: Annotated[
        str, click.option("--user", type=str, help="Db username", required=True)
    ]

    password: Annotated[str, click.option("--password", type=str, help="Db password")]

    index_name: Annotated[str, click.option("--index-name", type=str, help="Db index name", required=True)]

    filter_type: Annotated[
        str, click.option("--filter-type", type=str, help="post_filter|pre_filter|efficient_filter", required=False)]


class LindormHNSWTypedDict(CommonTypedDict, LindormTypedDict, HNSWFlavor3):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(LindormHNSWTypedDict)
def LindormHNSW(**parameters: Unpack[LindormHNSWTypedDict]):
    from .config import HNSWConfig, LindormConfig
    run(
        db=DB.Lindorm,
        db_config=LindormConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
            index_name=parameters["index_name"],
        ),
        db_case_config=HNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            efSearch=parameters["ef_search"],
            filter_type=parameters["filter_type"],
        ),
        **parameters,
    )

class LindormIVFBQTypedMinDict(CommonTypedDict, LindormTypedDict, IVFFlatTypedDictN):
    exbits: Annotated[
        str, click.option("--exbits",
                          type=int, help="Exbits",
                          required=True)
    ]

class LindormIVFPQTypedDict(CommonTypedDict, LindormTypedDict, IVFFlatTypedDictN, HNSWFlavor3):
    reorder_factor: Annotated[str, click.option("--reorder-factor", type=str, help="reorder factor", required=False)]

    client_refactor: Annotated[
        bool, click.option("--client-refactor", type=bool, help="client refactor", required=False)]

    k_expand_scope: Annotated[
        int, click.option("--k-expand-scope", type=int, help="k expand scope", required=False)
    ]

@cli.command()
@click_parameter_decorators_from_typed_dict(LindormIVFPQTypedDict)
def LindormIVFPQ(**parameters: Unpack[LindormIVFPQTypedDict]):
    from .config import IVFPQConfig, LindormConfig
    run(
        db=DB.Lindorm,
        db_config=LindormConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
            index_name=parameters["index_name"],
        ),
        db_case_config=IVFPQConfig(
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            centroids_hnsw_M=parameters["m"],
            centroids_hnsw_efConstruction=parameters["ef_construction"],
            centroids_hnsw_efSearch=parameters["ef_search"],
            filter_type=parameters["filter_type"],
            reorder_factor=parameters["reorder_factor"],
            client_refactor=parameters["client_refactor"],
            k_expand_scope=parameters["k_expand_scope"],
        ),
        **parameters,
    )

class LindormIVFBQTypedDict(CommonTypedDict, LindormTypedDict, LindormIVFBQTypedMinDict, HNSWFlavor3):
    reorder_factor: Annotated[str, click.option("--reorder-factor", type=str, help="reorder factor", required=False)]

    client_refactor: Annotated[
        bool, click.option("--client-refactor", type=bool, help="client refactor", required=False)]

    k_expand_scope: Annotated[
        int, click.option("--k-expand-scope", type=int, help="k expand scope", required=False)
    ]

@cli.command()
@click_parameter_decorators_from_typed_dict(LindormIVFBQTypedDict)
def LindormIVFBQ(**parameters: Unpack[LindormIVFBQTypedDict]):
    from .config import IVFBQConfig, LindormConfig
    run(
        db=DB.Lindorm,
        db_config=LindormConfig(
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
            index_name=parameters["index_name"],
        ),
        db_case_config=IVFBQConfig(
            nlist=parameters["nlist"],
            exbits=parameters["exbits"],
            nprobe=parameters["nprobe"],
            centroids_hnsw_M=parameters["m"],
            centroids_hnsw_efConstruction=parameters["ef_construction"],
            centroids_hnsw_efSearch=parameters["ef_search"],
            filter_type=parameters["filter_type"],
            reorder_factor=parameters["reorder_factor"],
            client_refactor=parameters["client_refactor"],
            k_expand_scope=parameters["k_expand_scope"],
        ),
        **parameters,
    )

