from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..api import IndexType


class LanceDBTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="URI connection string", required=True),
    ]
    token: Annotated[
        str | None,
        click.option("--token", type=str, help="Authentication token", required=False),
    ]
    cos_secret_id: Annotated[
        str | None,
        click.option(
            "--cos-secret-id",
            type=str,
            help="Tencent COS secret ID (or set COS_SECRET_ID env var)",
            required=False,
        ),
    ]
    cos_secret_key: Annotated[
        str | None,
        click.option(
            "--cos-secret-key",
            type=str,
            help="Tencent COS secret key (or set COS_SECRET_KEY env var)",
            required=False,
        ),
    ]
    cos_endpoint: Annotated[
        str | None,
        click.option(
            "--cos-endpoint",
            type=str,
            help="Tencent COS endpoint (or set COS_ENDPOINT env var)",
            required=False,
        ),
    ]
    cos_region: Annotated[
        str | None,
        click.option(
            "--cos-region",
            type=str,
            help="Tencent COS region (or set TENCENTCLOUD_REGION env var)",
            required=False,
        ),
    ]


def _build_db_config(**parameters):
    from .config import LanceDBConfig, build_lancedb_storage_options

    uri = parameters["uri"]
    return LanceDBConfig(
        db_label=parameters["db_label"],
        uri=uri,
        token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        storage_options=build_lancedb_storage_options(
            uri,
            cos_secret_id=parameters.get("cos_secret_id"),
            cos_secret_key=parameters.get("cos_secret_key"),
            cos_endpoint=parameters.get("cos_endpoint"),
            cos_region=parameters.get("cos_region"),
        ),
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDB(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBNoIndexConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(**parameters),
        db_case_config=LanceDBNoIndexConfig(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDBAutoIndex(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBAutoIndexConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(**parameters),
        db_case_config=LanceDBAutoIndexConfig(),
        **parameters,
    )


class LanceDBIVFPQTypedDict(CommonTypedDict, LanceDBTypedDict):
    num_partitions: Annotated[
        int,
        click.option(
            "--num-partitions",
            type=int,
            default=0,
            help="Number of partitions for IVF_PQ index, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    num_sub_vectors: Annotated[
        int,
        click.option(
            "--num-sub-vectors",
            type=int,
            default=0,
            help="Number of sub-vectors for IVF_PQ index, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=int,
            default=8,
            help="Number of bits for quantization (4 or 8)",
            show_default=True,
        ),
    ]
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes",
            type=int,
            default=0,
            help="Number of probes for IVF search, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    refine_factor: Annotated[
        int,
        click.option(
            "--refine-factor",
            type=int,
            default=0,
            help="Refine factor for better recall, 0 = disabled",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFPQTypedDict)
def LanceDBIVFPQ(**parameters: Unpack[LanceDBIVFPQTypedDict]):
    from .config import LanceDBIndexConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(**parameters),
        db_case_config=LanceDBIndexConfig(
            index=IndexType.IVFPQ,
            num_partitions=parameters["num_partitions"],
            num_sub_vectors=parameters["num_sub_vectors"],
            nbits=parameters["nbits"],
            nprobes=parameters["nprobes"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class LanceDBIVFHNSWSQTypedDict(CommonTypedDict, LanceDBTypedDict):
    num_partitions: Annotated[
        int,
        click.option(
            "--num-partitions",
            type=int,
            default=0,
            help="Number of IVF partitions, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, default=0, help="HNSW parameter m, 0 = use LanceDB default", show_default=True),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            default=0,
            help="HNSW ef_construction, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    ef: Annotated[
        int,
        click.option("--ef", type=int, default=0, help="HNSW search ef, 0 = use LanceDB default", show_default=True),
    ]
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes",
            type=int,
            default=0,
            help="Number of probes for IVF search, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    refine_factor: Annotated[
        int,
        click.option(
            "--refine-factor",
            type=int,
            default=0,
            help="Refine factor for better recall, 0 = disabled",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFHNSWSQTypedDict)
def LanceDBIVFHNSWSQ(**parameters: Unpack[LanceDBIVFHNSWSQTypedDict]):
    from .config import LanceDBIVFHNSWSQConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(**parameters),
        db_case_config=LanceDBIVFHNSWSQConfig(
            num_partitions=parameters["num_partitions"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef=parameters["ef"],
            nprobes=parameters["nprobes"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class LanceDBIVFHNSWPQTypedDict(CommonTypedDict, LanceDBTypedDict):
    num_partitions: Annotated[
        int,
        click.option(
            "--num-partitions",
            type=int,
            default=0,
            help="Number of IVF partitions, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    num_sub_vectors: Annotated[
        int,
        click.option(
            "--num-sub-vectors",
            type=int,
            default=0,
            help="Number of PQ sub-vectors, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, default=0, help="HNSW parameter m, 0 = use LanceDB default", show_default=True),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            default=0,
            help="HNSW ef_construction, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    ef: Annotated[
        int,
        click.option("--ef", type=int, default=0, help="HNSW search ef, 0 = use LanceDB default", show_default=True),
    ]
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes",
            type=int,
            default=0,
            help="Number of probes for IVF search, 0 = use LanceDB default",
            show_default=True,
        ),
    ]
    refine_factor: Annotated[
        int,
        click.option(
            "--refine-factor",
            type=int,
            default=0,
            help="Refine factor for better recall, 0 = disabled",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFHNSWPQTypedDict)
def LanceDBIVFHNSWPQ(**parameters: Unpack[LanceDBIVFHNSWPQTypedDict]):
    from .config import LanceDBIVFHNSWPQConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(**parameters),
        db_case_config=LanceDBIVFHNSWPQConfig(
            num_partitions=parameters["num_partitions"],
            num_sub_vectors=parameters["num_sub_vectors"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef=parameters["ef"],
            nprobes=parameters["nprobes"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )
