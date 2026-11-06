from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from .config import DEFAULT_WARM_PATH


class WarmTypedDict(CommonTypedDict):
    path: Annotated[
        str,
        click.option("--path", type=str, default=DEFAULT_WARM_PATH, show_default=True, help="index directory"),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, default=16, show_default=True, help="HNSW M; WARM degree = 2*M"),
    ]
    ef_construction: Annotated[
        int,
        click.option("--ef-construction", type=int, default=300, show_default=True, help="WARM build_ef"),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, default=128, show_default=True, help="WARM candidate pool size ef"),
    ]
    rerank: Annotated[
        int,
        click.option(
            "--rerank",
            type=int,
            default=0,
            show_default=True,
            help="pool candidates re-scored with the residual code; 0 means max(ef, k)",
        ),
    ]
    code_bits: Annotated[
        int,
        click.option("--code-bits", type=click.Choice(["4", "8"]), default="4", help="traversal code width"),
    ]
    residual_bits: Annotated[
        int,
        click.option("--residual-bits", type=click.Choice(["0", "4", "8"]), default="8", help="residual width"),
    ]
    threads: Annotated[
        int,
        click.option("--threads", type=int, default=0, show_default=True, help="build threads (0 = one per core)"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(WarmTypedDict)
def Warm(**parameters: Unpack[WarmTypedDict]):
    from .config import WarmConfig, WarmIndexConfig

    run(
        db=DB.Warm,
        db_config=WarmConfig(db_label=parameters["db_label"], path=parameters["path"]),
        db_case_config=WarmIndexConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            rerank=parameters["rerank"],
            code_bits=int(parameters["code_bits"]),
            residual_bits=int(parameters["residual_bits"]),
            threads=parameters["threads"],
        ),
        **parameters,
    )
