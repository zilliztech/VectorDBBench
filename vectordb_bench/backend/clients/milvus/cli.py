from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    IVFFlatTypedDictN,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,

)
from vectordb_bench.backend.clients import DB

DBTYPE = DB.Milvus


class MilvusTypedDict(TypedDict):
    uri: Annotated[
        str, click.option("--uri", type=str, help="uri connection string", required=True)
    ]


class MilvusAutoIndexTypedDict(CommonTypedDict, MilvusTypedDict):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusAutoIndexTypedDict)
def MilvusAutoIndex(**parameters: Unpack[MilvusAutoIndexTypedDict]):
    from .config import MilvusConfig, AutoIndexConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=AutoIndexConfig(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusAutoIndexTypedDict)
def MilvusFlat(**parameters: Unpack[MilvusAutoIndexTypedDict]):
    from .config import MilvusConfig, FLATConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=FLATConfig(),
        **parameters,
    )


class MilvusHNSWTypedDict(CommonTypedDict, MilvusTypedDict, HNSWFlavor3):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusHNSWTypedDict)
def MilvusHNSW(**parameters: Unpack[MilvusHNSWTypedDict]):
    from .config import MilvusConfig, HNSWConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=HNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_search"],
        ),
        **parameters,
    )


class MilvusIVFFlatTypedDict(CommonTypedDict, MilvusTypedDict, IVFFlatTypedDictN):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusIVFFlatTypedDict)
def MilvusIVFFlat(**parameters: Unpack[MilvusIVFFlatTypedDict]):
    from .config import MilvusConfig, IVFFlatConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=IVFFlatConfig(
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusIVFFlatTypedDict)
def MilvusIVFSQ8(**parameters: Unpack[MilvusIVFFlatTypedDict]):
    from .config import MilvusConfig, IVFSQ8Config

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=IVFSQ8Config(
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
        ),
        **parameters,
    )


class MilvusDISKANNTypedDict(CommonTypedDict, MilvusTypedDict):
    search_list: Annotated[
        str, click.option("--search-list",
                          type=int,
                          required=True)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusDISKANNTypedDict)
def MilvusDISKANN(**parameters: Unpack[MilvusDISKANNTypedDict]):
    from .config import MilvusConfig, DISKANNConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=DISKANNConfig(
            search_list=parameters["search_list"],
        ),
        **parameters,
    )


class MilvusGPUIVFTypedDict(CommonTypedDict, MilvusTypedDict, MilvusIVFFlatTypedDict):
    cache_dataset_on_device: Annotated[
        str, click.option("--cache-dataset-on-device",
                          type=str,
                          required=True)
    ]
    refine_ratio: Annotated[
        str, click.option("--refine-ratio",
                          type=float,
                          required=True)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUIVFTypedDict)
def MilvusGPUIVFFlat(**parameters: Unpack[MilvusGPUIVFTypedDict]):
    from .config import MilvusConfig, GPUIVFFlatConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=GPUIVFFlatConfig(
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            cache_dataset_on_device=parameters["cache_dataset_on_device"],
            refine_ratio=parameters.get("refine_ratio"),
        ),
        **parameters,
    )


class MilvusGPUIVFPQTypedDict(CommonTypedDict, MilvusTypedDict, MilvusIVFFlatTypedDict, MilvusGPUIVFTypedDict):
    m: Annotated[
        str, click.option("--m",
                          type=int, help="hnsw m",
                          required=True)
    ]
    nbits: Annotated[
        str, click.option("--nbits",
                          type=int,
                          required=True)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUIVFPQTypedDict)
def MilvusGPUIVFPQ(**parameters: Unpack[MilvusGPUIVFPQTypedDict]):
    from .config import MilvusConfig, GPUIVFPQConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=GPUIVFPQConfig(
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            m=parameters["m"],
            nbits=parameters["nbits"],
            cache_dataset_on_device=parameters["cache_dataset_on_device"],
            refine_ratio=parameters["refine_ratio"],
        ),
        **parameters,
    )


class MilvusGPUCAGRATypedDict(CommonTypedDict, MilvusTypedDict, MilvusGPUIVFTypedDict):
    intermediate_graph_degree: Annotated[
        str, click.option("--intermediate-graph-degree",
                          type=int,
                          required=True)
    ]
    graph_degree: Annotated[
        str, click.option("--graph-degree",
                          type=int,
                          required=True)
    ]
    build_algo: Annotated[
        str, click.option("--build_algo",
                          type=str,
                          required=True)
    ]
    team_size: Annotated[
        str, click.option("--team-size",
                          type=int,
                          required=True)
    ]
    search_width: Annotated[
        str, click.option("--search-width",
                          type=int,
                          required=True)
    ]
    itopk_size: Annotated[
        str, click.option("--itopk-size",
                          type=int,
                          required=True)
    ]
    min_iterations: Annotated[
        str, click.option("--min-iterations",
                          type=int,
                          required=True)
    ]
    max_iterations: Annotated[
        str, click.option("--max-iterations",
                          type=int,
                          required=True)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUCAGRATypedDict)
def MilvusGPUCAGRA(**parameters: Unpack[MilvusGPUCAGRATypedDict]):
    from .config import MilvusConfig, GPUCAGRAConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
        ),
        db_case_config=GPUCAGRAConfig(
            intermediate_graph_degree=parameters["intermediate_graph_degree"],
            graph_degree=parameters["graph_degree"],
            itopk_size=parameters["itopk_size"],
            team_size=parameters["team_size"],
            search_width=parameters["search_width"],
            min_iterations=parameters["min_iterations"],
            max_iterations=parameters["max_iterations"],
            build_algo=parameters["build_algo"],
            cache_dataset_on_device=parameters["cache_dataset_on_device"],
            refine_ratio=parameters["refine_ratio"],
        ),
        **parameters,
    )
