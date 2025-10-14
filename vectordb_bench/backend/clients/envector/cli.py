from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.EnVector


class EnVectorTypedDict(TypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    eval_mode: Annotated[
        str,
        click.option("--eval-mode", help="Evaluation mode", type=click.Choice(["mm", "rmp"]), default="mm"),
    ]
    

class EnVectorFlatIndexTypedDict(CommonTypedDict, EnVectorTypedDict): ...


@cli.command(name="envectorflat")
@click_parameter_decorators_from_typed_dict(EnVectorFlatIndexTypedDict)
def EnVectorFlat(**parameters: Unpack[EnVectorFlatIndexTypedDict]):
    from .config import FlatIndexConfig, EnVectorConfig

    run(
        db=DBTYPE,
        db_config=EnVectorConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            eval_mode=parameters["eval_mode"],
            index_params={},
        ),
        db_case_config=FlatIndexConfig(),
        **parameters,
    )


class EnVectorIVFFlatIndexTypedDict(CommonTypedDict, EnVectorTypedDict): 
    nlist: Annotated[
        int,
        click.option("--nlist", type=int, help="nlist for IVF index", default=250),
    ]
    nprobe: Annotated[
        int,
        click.option("--nprobe", type=int, help="nprobe for IVF index", default=6),
    ]


@cli.command(name="envectorivfflat")
@click_parameter_decorators_from_typed_dict(EnVectorIVFFlatIndexTypedDict)
def EnVectorIVFFlat(**parameters: Unpack[EnVectorIVFFlatIndexTypedDict]):
    from .config import IVFFlatIndexConfig, EnVectorConfig

    run(
        db=DBTYPE,
        db_config=EnVectorConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            eval_mode=parameters["eval_mode"],
            index_params={"nlist": parameters["nlist"], "nprobe": parameters["nprobe"]},
        ),
        db_case_config=IVFFlatIndexConfig(nlist=parameters["nlist"], nprobe=parameters["nprobe"]),
        **parameters,
    )
