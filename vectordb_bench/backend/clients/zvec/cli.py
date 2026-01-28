from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class ZvecTypedDict(CommonTypedDict):
    path: Annotated[
        str,
        click.option("--path", type=str, help="collection path", required=True),
    ]


class ZvecHNSWTypedDict(CommonTypedDict, ZvecTypedDict):
    m: Annotated[
        int,
        click.option("--m", type=int, default=50, help="HNSW index parameter m."),
    ]
    ef_construct: Annotated[
        int,
        click.option("--ef-construction", type=int, default=500, help="HNSW index parameter ef_construction"),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, default=300, help="HNSW index parameter ef for search"),
    ]
    quantize_type: Annotated[
        int,
        click.option("--quantize-type", type=str, default="", help="HNSW index quantize type, fp16/int8 supported"),
    ]
    is_using_refiner: Annotated[
        bool,
        click.option(
            "--is-using-refiner",
            is_flag=True,
            default=False,
            help="is using refiner, suitable for quantized index, "
            "recall `ef-search` results then refine with unquantized vector to `topk` results",
        ),
    ]


# default to hnsw
@cli.command()
@click_parameter_decorators_from_typed_dict(ZvecHNSWTypedDict)
def Zvec(**parameters: Unpack[ZvecHNSWTypedDict]):
    from .config import ZvecConfig, ZvecHNSWIndexConfig

    run(
        db=DB.Zvec,
        db_config=ZvecConfig(
            db_label=parameters["db_label"],
            path=parameters["path"],
        ),
        db_case_config=ZvecHNSWIndexConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            quantize_type=parameters["quantize_type"],
            is_using_refiner=parameters["is_using_refiner"],
        ),
        **parameters,
    )
