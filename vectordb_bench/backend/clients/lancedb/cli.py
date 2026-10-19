import os
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


def _build_storage_options(**parameters) -> dict[str, str] | None:
    """Build storage_options based on URI scheme.

    Supports:
    - cos:// / s3://  → Tencent COS credentials
    - goosefs://      → GooseFS authentication options
    """
    uri = parameters.get("uri", "")

    # --- GooseFS storage options ---
    if uri.startswith("goosefs://"):
        return _build_goosefs_storage_options(**parameters)

    # --- COS / S3 storage options ---
    if uri.startswith(("cos://", "s3://")):
        return _build_cos_storage_options(**parameters)

    return None


def _build_cos_storage_options(**parameters) -> dict[str, str] | None:
    """Build storage_options for COS/S3 if credentials are provided."""
    secret_id = parameters.get("cos_secret_id") or os.environ.get("COS_SECRET_ID")
    secret_key = parameters.get("cos_secret_key") or os.environ.get("COS_SECRET_KEY")
    endpoint = parameters.get("cos_endpoint") or os.environ.get("COS_ENDPOINT")
    region = parameters.get("cos_region") or os.environ.get("TENCENTCLOUD_REGION")

    if not (secret_id and secret_key):
        return None

    storage_options = {
        "aws_access_key_id": secret_id,
        "aws_secret_access_key": secret_key,
    }
    if endpoint:
        storage_options["endpoint"] = endpoint
    if region:
        storage_options["region"] = region

    return storage_options


def _build_goosefs_storage_options(**_parameters: str) -> dict[str, str] | None:
    """Build storage_options for GooseFS.

    Recognized environment variables:
    - GOOSEFS_AUTH_TYPE       → goosefs_auth_type (simple / nosasl)
    - GOOSEFS_AUTH_USERNAME   → goosefs_auth_username
    - GOOSEFS_WRITE_TYPE      → goosefs_write_type (CACHE_THROUGH etc.)
    - GOOSEFS_BLOCK_SIZE      → goosefs_block_size
    - GOOSEFS_CHUNK_SIZE      → goosefs_chunk_size
    """
    storage_options: dict[str, str] = {}

    _goosefs_env_keys = {
        "GOOSEFS_AUTH_TYPE": "goosefs_auth_type",
        "GOOSEFS_AUTH_USERNAME": "goosefs_auth_username",
        "GOOSEFS_WRITE_TYPE": "goosefs_write_type",
        "GOOSEFS_BLOCK_SIZE": "goosefs_block_size",
        "GOOSEFS_CHUNK_SIZE": "goosefs_chunk_size",
    }

    for env_key, opt_key in _goosefs_env_keys.items():
        value = os.environ.get(env_key)
        if value:
            storage_options[opt_key] = value

    return storage_options if storage_options else None


def _build_db_config(**parameters):
    from .config import LanceDBConfig

    return LanceDBConfig(
        db_label=parameters["db_label"],
        uri=parameters["uri"],
        token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        storage_options=_build_storage_options(**parameters),
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
