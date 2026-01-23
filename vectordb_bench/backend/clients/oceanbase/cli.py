import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor4,
    OceanBaseIVFTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

from ..api import IndexType


class OceanBaseTypedDict(CommonTypedDict):
    host: Annotated[str, click.option("--host", type=str, help="OceanBase host", default="")]
    user: Annotated[str, click.option("--user", type=str, help="OceanBase username", required=True)]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="OceanBase database password",
            default=lambda: os.environ.get("OB_PASSWORD", ""),
        ),
    ]
    database: Annotated[str, click.option("--database", type=str, help="DataBase name", required=True)]
    port: Annotated[int, click.option("--port", type=int, help="OceanBase port", required=True)]


class OceanBaseHNSWTypedDict(CommonTypedDict, OceanBaseTypedDict, HNSWFlavor4): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(OceanBaseHNSWTypedDict)
def OceanBaseHNSW(**parameters: Unpack[OceanBaseHNSWTypedDict]):
    from .config import OceanBaseConfig, OceanBaseHNSWConfig

    run(
        db=DB.OceanBase,
        db_config=OceanBaseConfig(
            db_label=parameters["db_label"],
            user=SecretStr(parameters["user"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
        ),
        db_case_config=OceanBaseHNSWConfig(
            m=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            index=parameters["index_type"],
        ),
        **parameters,
    )


class OceanBaseIVFTypedDict(CommonTypedDict, OceanBaseTypedDict, OceanBaseIVFTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(OceanBaseIVFTypedDict)
def OceanBaseIVF(**parameters: Unpack[OceanBaseIVFTypedDict]):
    from .config import OceanBaseConfig, OceanBaseIVFConfig

    type_str = parameters["index_type"]
    if type_str == "IVF_FLAT":
        input_index_type = IndexType.IVFFlat
    elif type_str == "IVF_PQ":
        input_index_type = IndexType.IVFPQ
    elif type_str == "IVF_SQ8":
        input_index_type = IndexType.IVFSQ8

    input_m = 0 if parameters["m"] is None else parameters["m"]

    run(
        db=DB.OceanBase,
        db_config=OceanBaseConfig(
            db_label=parameters["db_label"],
            user=SecretStr(parameters["user"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
        ),
        db_case_config=OceanBaseIVFConfig(
            m=input_m,
            nlist=parameters["nlist"],
            sample_per_nlist=parameters["sample_per_nlist"],
            nbits=parameters["nbits"],
            index=input_index_type,
            ivf_nprobes=parameters["ivf_nprobes"],
        ),
        **parameters,
    )
