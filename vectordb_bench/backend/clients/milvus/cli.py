from typing import Annotated, TypedDict, Unpack

import click
from pydantic import BaseModel, SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    IVFFlatTypedDictN,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.Milvus


def _use_partition_key(parameters: dict) -> bool:
    explicit = parameters.get("use_partition_key")
    if explicit is not None:
        return explicit
    return parameters.get("case_type") == "CloudMultiTenantSearchCase"


def _with_partition_key(db_case_config: BaseModel, parameters: dict) -> BaseModel:
    return db_case_config.model_copy(update={"use_partition_key": _use_partition_key(parameters)})


class MilvusTypedDict(TypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    user_name: Annotated[
        str | None,
        click.option("--user-name", type=str, help="Db username", required=False),
    ]
    password: Annotated[
        str | None,
        click.option("--password", type=str, help="Db password", required=False),
    ]
    num_shards: Annotated[
        int,
        click.option(
            "--num-shards",
            type=int,
            help="Number of shards",
            required=False,
            default=1,
            show_default=True,
        ),
    ]
    replica_number: Annotated[
        int,
        click.option(
            "--replica-number",
            type=int,
            help="Number of replicas",
            required=False,
            default=1,
            show_default=True,
        ),
    ]
    use_partition_key: Annotated[
        bool | None,
        click.option(
            "--use-partition-key/--no-use-partition-key",
            default=None,
            help=(
                "Use the Milvus partition key on the label field. "
                "Defaults to enabled for CloudMultiTenantSearchCase and disabled otherwise."
            ),
        ),
    ]


class MilvusAutoIndexTypedDict(CommonTypedDict, MilvusTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusAutoIndexTypedDict)
def MilvusAutoIndex(**parameters: Unpack[MilvusAutoIndexTypedDict]):
    from .config import AutoIndexConfig, MilvusConfig

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
        db_case_config=_with_partition_key(AutoIndexConfig(), parameters),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusAutoIndexTypedDict)
def MilvusFlat(**parameters: Unpack[MilvusAutoIndexTypedDict]):
    from .config import FLATConfig, MilvusConfig

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
        db_case_config=_with_partition_key(FLATConfig(), parameters),
        **parameters,
    )


class MilvusHNSWTypedDict(CommonTypedDict, MilvusTypedDict, HNSWFlavor3): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusHNSWTypedDict)
def MilvusHNSW(**parameters: Unpack[MilvusHNSWTypedDict]):
    from .config import HNSWConfig, MilvusConfig

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
            HNSWConfig(
                M=parameters["m"],
                efConstruction=parameters["ef_construction"],
                ef=parameters["ef_search"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusRefineTypedDict(TypedDict):
    refine: Annotated[
        bool,
        click.option(
            "--refine",
            type=bool,
            required=True,
            help="Whether refined data is reserved during index building.",
        ),
    ]
    refine_type: Annotated[
        str | None,
        click.option(
            "--refine-type",
            type=click.Choice(["SQ6", "SQ8", "BF16", "FP16", "FP32"], case_sensitive=False),
            help="The data type of the refine index to use. Supported values: SQ6,SQ8,BF16,FP16,FP32",
            required=True,
        ),
    ]
    refine_k: Annotated[
        float,
        click.option(
            "--refine-k",
            type=float,
            help="The magnification factor of refine compared to k.",
            required=True,
        ),
    ]


class MilvusHNSWPQTypedDict(CommonTypedDict, MilvusTypedDict, MilvusHNSWTypedDict, MilvusRefineTypedDict):
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=int,
            required=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusHNSWPQTypedDict)
def MilvusHNSWPQ(**parameters: Unpack[MilvusHNSWPQTypedDict]):
    from .config import HNSWPQConfig, MilvusConfig

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
            HNSWPQConfig(
                M=parameters["m"],
                efConstruction=parameters["ef_construction"],
                ef=parameters["ef_search"],
                nbits=parameters["nbits"],
                refine=parameters["refine"],
                refine_type=parameters["refine_type"],
                refine_k=parameters["refine_k"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusHNSWPRQTypedDict(
    CommonTypedDict,
    MilvusTypedDict,
    MilvusHNSWPQTypedDict,
):
    nrq: Annotated[
        int,
        click.option(
            "--nrq",
            type=int,
            help="The number of residual subquantizers.",
            required=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusHNSWPRQTypedDict)
def MilvusHNSWPRQ(**parameters: Unpack[MilvusHNSWPRQTypedDict]):
    from .config import HNSWPRQConfig, MilvusConfig

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
            HNSWPRQConfig(
                M=parameters["m"],
                efConstruction=parameters["ef_construction"],
                ef=parameters["ef_search"],
                nbits=parameters["nbits"],
                refine=parameters["refine"],
                refine_type=parameters["refine_type"],
                refine_k=parameters["refine_k"],
                nrq=parameters["nrq"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusHNSWSQTypedDict(CommonTypedDict, MilvusTypedDict, MilvusHNSWTypedDict, MilvusRefineTypedDict):
    sq_type: Annotated[
        str | None,
        click.option(
            "--sq-type",
            type=click.Choice(["SQ4U", "SQ6", "SQ8", "BF16", "FP16", "FP32"], case_sensitive=False),
            help="Scalar quantizer type. Supported values: SQ4U,SQ6,SQ8,BF16,FP16,FP32",
            required=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusHNSWSQTypedDict)
def MilvusHNSWSQ(**parameters: Unpack[MilvusHNSWSQTypedDict]):
    from .config import HNSWSQConfig, MilvusConfig

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
            HNSWSQConfig(
                M=parameters["m"],
                efConstruction=parameters["ef_construction"],
                ef=parameters["ef_search"],
                sq_type=parameters["sq_type"],
                refine=parameters["refine"],
                refine_type=parameters["refine_type"],
                refine_k=parameters["refine_k"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusIVFFlatTypedDict(CommonTypedDict, MilvusTypedDict, IVFFlatTypedDictN): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusIVFFlatTypedDict)
def MilvusIVFFlat(**parameters: Unpack[MilvusIVFFlatTypedDict]):
    from .config import IVFFlatConfig, MilvusConfig

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
            IVFFlatConfig(
                nlist=parameters["nlist"],
                nprobe=parameters["nprobe"],
            ),
            parameters,
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusIVFFlatTypedDict)
def MilvusIVFSQ8(**parameters: Unpack[MilvusIVFFlatTypedDict]):
    from .config import IVFSQ8Config, MilvusConfig

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
            IVFSQ8Config(
                nlist=parameters["nlist"],
                nprobe=parameters["nprobe"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusIVFRABITQTypedDict(CommonTypedDict, MilvusTypedDict, MilvusIVFFlatTypedDict):
    rbq_bits_query: Annotated[
        int,
        click.option(
            "--rbq-bits-query",
            type=int,
            help="The level of quantization of a query vector. Use 1…8 for the SQ1…SQ8 and 0 to disable.",
            required=True,
        ),
    ]
    refine: Annotated[
        bool,
        click.option(
            "--refine",
            type=bool,
            required=True,
            help="Whether refined data is reserved during index building.",
        ),
    ]
    refine_type: Annotated[
        str | None,
        click.option(
            "--refine-type",
            type=click.Choice(["SQ6", "SQ8", "BF16", "FP16", "FP32"], case_sensitive=False),
            help="The data type of the refine index to use. Supported values: SQ6,SQ8,BF16,FP16,FP32",
            required=True,
        ),
    ]
    refine_k: Annotated[
        float,
        click.option(
            "--refine-k",
            type=float,
            help="The magnification factor of refine compared to k.",
            required=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusIVFRABITQTypedDict)
def MilvusIVFRabitQ(**parameters: Unpack[MilvusIVFRABITQTypedDict]):
    from .config import IVFRABITQConfig, MilvusConfig

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
            IVFRABITQConfig(
                nlist=parameters["nlist"],
                nprobe=parameters["nprobe"],
                rbq_bits_query=parameters["rbq_bits_query"],
                refine=parameters["refine"],
                refine_type=parameters["refine_type"],
                refine_k=parameters["refine_k"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusDISKANNTypedDict(CommonTypedDict, MilvusTypedDict):
    search_list: Annotated[str, click.option("--search-list", type=int, required=True)]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusDISKANNTypedDict)
def MilvusDISKANN(**parameters: Unpack[MilvusDISKANNTypedDict]):
    from .config import DISKANNConfig, MilvusConfig

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
            DISKANNConfig(
                search_list=parameters["search_list"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusGPUIVFTypedDict(CommonTypedDict, MilvusTypedDict, MilvusIVFFlatTypedDict):
    cache_dataset_on_device: Annotated[
        str,
        click.option("--cache-dataset-on-device", type=str, required=True),
    ]
    refine_ratio: Annotated[str, click.option("--refine-ratio", type=float, required=True)]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUIVFTypedDict)
def MilvusGPUIVFFlat(**parameters: Unpack[MilvusGPUIVFTypedDict]):
    from .config import GPUIVFFlatConfig, MilvusConfig

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
            GPUIVFFlatConfig(
                nlist=parameters["nlist"],
                nprobe=parameters["nprobe"],
                cache_dataset_on_device=parameters["cache_dataset_on_device"],
                refine_ratio=parameters.get("refine_ratio"),
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusGPUBruteForceTypedDict(CommonTypedDict, MilvusTypedDict):
    metric_type: Annotated[
        str,
        click.option("--metric-type", type=str, required=True, help="Metric type for brute force search"),
    ]
    limit: Annotated[
        int,
        click.option("--limit", type=int, required=True, help="Top-k limit for search"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUBruteForceTypedDict)
def MilvusGPUBruteForce(**parameters: Unpack[MilvusGPUBruteForceTypedDict]):
    from .config import GPUBruteForceConfig, MilvusConfig

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
            GPUBruteForceConfig(
                metric_type=parameters["metric_type"],
                limit=parameters["limit"],  # top-k for search
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusSVSVamanaTypedDict(CommonTypedDict, MilvusTypedDict):
    svs_graph_max_degree: Annotated[
        int,
        click.option(
            "--svs-graph-max-degree",
            type=int,
            help="Maximum degree of the Vamana graph (4-256).",
            required=True,
        ),
    ]
    svs_construction_window_size: Annotated[
        int,
        click.option(
            "--svs-construction-window-size",
            type=int,
            help="Window size for graph construction.",
            required=False,
            default=40,
            show_default=True,
        ),
    ]
    svs_alpha: Annotated[
        float | None,
        click.option(
            "--svs-alpha",
            type=float,
            help="Pruning parameter (default: 1.2 for L2, 0.95 for IP/COSINE).",
            required=False,
            default=None,
        ),
    ]
    svs_storage_kind: Annotated[
        str,
        click.option(
            "--svs-storage-kind",
            type=click.Choice(
                ["fp32", "fp16", "sqi8", "lvq4x0", "lvq4x4", "lvq4x8", "leanvec4x4", "leanvec4x8", "leanvec8x8"],
                case_sensitive=False,
            ),
            help="Data storage format.",
            required=False,
            default="fp32",
            show_default=True,
        ),
    ]
    svs_search_window_size: Annotated[
        int | None,
        click.option(
            "--svs-search-window-size",
            type=int,
            help="Window size for search (1-10000).",
            required=False,
            default=None,
        ),
    ]
    svs_search_buffer_capacity: Annotated[
        int | None,
        click.option(
            "--svs-search-buffer-capacity",
            type=int,
            help="Buffer capacity for search priority queue (1-10000).",
            required=False,
            default=None,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusSVSVamanaTypedDict)
def MilvusSVSVamana(**parameters: Unpack[MilvusSVSVamanaTypedDict]):
    from .config import MilvusConfig, SVSVamanaConfig

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
            SVSVamanaConfig(
                svs_graph_max_degree=parameters["svs_graph_max_degree"],
                svs_construction_window_size=parameters["svs_construction_window_size"],
                svs_alpha=parameters["svs_alpha"],
                svs_storage_kind=parameters["svs_storage_kind"],
                svs_search_window_size=parameters["svs_search_window_size"],
                svs_search_buffer_capacity=parameters["svs_search_buffer_capacity"],
            ),
            parameters,
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusSVSVamanaTypedDict)
def MilvusSVSVamanaLVQ(**parameters: Unpack[MilvusSVSVamanaTypedDict]):
    from .config import MilvusConfig, SVSVamanaLVQConfig

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
            SVSVamanaLVQConfig(
                svs_graph_max_degree=parameters["svs_graph_max_degree"],
                svs_construction_window_size=parameters["svs_construction_window_size"],
                svs_alpha=parameters["svs_alpha"],
                svs_storage_kind=parameters["svs_storage_kind"],
                svs_search_window_size=parameters["svs_search_window_size"],
                svs_search_buffer_capacity=parameters["svs_search_buffer_capacity"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusSVSVamanaLeanVecTypedDict(MilvusSVSVamanaTypedDict):
    svs_leanvec_dim: Annotated[
        int,
        click.option(
            "--svs-leanvec-dim",
            type=int,
            help="Dimensionality for LeanVec compression (0 = d/2).",
            required=False,
            default=0,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusSVSVamanaLeanVecTypedDict)
def MilvusSVSVamanaLeanVec(**parameters: Unpack[MilvusSVSVamanaLeanVecTypedDict]):
    from .config import MilvusConfig, SVSVamanaLeanVecConfig

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
            SVSVamanaLeanVecConfig(
                svs_graph_max_degree=parameters["svs_graph_max_degree"],
                svs_construction_window_size=parameters["svs_construction_window_size"],
                svs_alpha=parameters["svs_alpha"],
                svs_storage_kind=parameters["svs_storage_kind"],
                svs_search_window_size=parameters["svs_search_window_size"],
                svs_search_buffer_capacity=parameters["svs_search_buffer_capacity"],
                svs_leanvec_dim=parameters["svs_leanvec_dim"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusGPUIVFPQTypedDict(
    CommonTypedDict,
    MilvusTypedDict,
    MilvusIVFFlatTypedDict,
    MilvusGPUIVFTypedDict,
):
    m: Annotated[str, click.option("--m", type=int, help="hnsw m", required=True)]
    nbits: Annotated[str, click.option("--nbits", type=int, required=True)]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUIVFPQTypedDict)
def MilvusGPUIVFPQ(**parameters: Unpack[MilvusGPUIVFPQTypedDict]):
    from .config import GPUIVFPQConfig, MilvusConfig

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
            GPUIVFPQConfig(
                nlist=parameters["nlist"],
                nprobe=parameters["nprobe"],
                m=parameters["m"],
                nbits=parameters["nbits"],
                cache_dataset_on_device=parameters["cache_dataset_on_device"],
                refine_ratio=parameters["refine_ratio"],
            ),
            parameters,
        ),
        **parameters,
    )


class MilvusGPUCAGRATypedDict(CommonTypedDict, MilvusTypedDict, MilvusGPUIVFTypedDict):
    intermediate_graph_degree: Annotated[
        str,
        click.option("--intermediate-graph-degree", type=int, required=True),
    ]
    graph_degree: Annotated[str, click.option("--graph-degree", type=int, required=True)]
    build_algo: Annotated[str, click.option("--build_algo", type=str, required=True)]
    team_size: Annotated[str, click.option("--team-size", type=int, required=True)]
    search_width: Annotated[str, click.option("--search-width", type=int, required=True)]
    itopk_size: Annotated[str, click.option("--itopk-size", type=int, required=True)]
    min_iterations: Annotated[str, click.option("--min-iterations", type=int, required=True)]
    max_iterations: Annotated[str, click.option("--max-iterations", type=int, required=True)]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusGPUCAGRATypedDict)
def MilvusGPUCAGRA(**parameters: Unpack[MilvusGPUCAGRATypedDict]):
    from .config import GPUCAGRAConfig, MilvusConfig

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
            GPUCAGRAConfig(
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
            parameters,
        ),
        **parameters,
    )


class MilvusFTSTypedDict(CommonTypedDict, MilvusTypedDict):
    """TypedDict for Milvus FTS command parameters."""

    bm25_k1: Annotated[
        float | None,
        click.option(
            "--bm25-k1",
            type=float,
            help="BM25 k1. Omit to use the Milvus product default.",
            required=False,
            default=None,
        ),
    ]
    bm25_b: Annotated[
        float | None,
        click.option(
            "--bm25-b",
            type=float,
            help="BM25 b. Omit to use the Milvus product default.",
            required=False,
            default=None,
        ),
    ]
    drop_ratio_search: Annotated[
        float | None,
        click.option(
            "--drop-ratio-search",
            type=float,
            help="Drop ratio for search (optional, for performance tuning)",
            required=False,
            default=None,
        ),
    ]
    use_force_merge: Annotated[
        bool,
        click.option(
            "--use-force-merge/--no-use-force-merge",
            type=bool,
            help="Compact sealed Milvus segments before FTS search.",
            required=False,
            default=True,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusFTSTypedDict)
def MilvusFTS(**parameters: Unpack[MilvusFTSTypedDict]):
    """Run FTS (Full-Text Search) benchmark on Milvus using BM25.

    This command uses the MS MARCO dev/small dataset for FTS testing.
    """
    from .config import MilvusConfig, MilvusFtsConfig

    # Set default case_type to large dataset if not specified
    if parameters.get("case_type") == "Performance1536D50K":  # Default from CommonTypedDict
        parameters["case_type"] = "FTSmsmarcoPerformance"

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
        db_case_config=MilvusFtsConfig(
            bm25_k1=parameters.get("bm25_k1"),
            bm25_b=parameters.get("bm25_b"),
            drop_ratio_search=parameters.get("drop_ratio_search"),
            use_force_merge=parameters["use_force_merge"],
        ),
        **parameters,
    )
