from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.Infino


class InfinoTypedDict(CommonTypedDict):
    data_path: Annotated[
        str,
        click.option("--data-path", type=str, default="/tmp/vectordb_bench/infino", help="Infino catalog directory"),
    ]
    table_name: Annotated[
        str,
        click.option("--table-name", type=str, default="vdbbench_infino", help="Infino table name"),
    ]
    n_cent: Annotated[int, click.option("--n-cent", type=int, default=256, help="IVF centroid count (build)")]
    nprobe: Annotated[int, click.option("--nprobe", type=int, default=32, help="IVF cells probed (query)")]


class InfinoFTSTypedDict(CommonTypedDict):
    data_path: Annotated[
        str,
        click.option("--data-path", type=str, default="/tmp/vectordb_bench/infino", help="Infino catalog directory"),
    ]
    table_name: Annotated[
        str,
        click.option("--table-name", type=str, default="vdbbench_infino_fts", help="Infino table name"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(InfinoTypedDict)
def Infino(**parameters: Unpack[InfinoTypedDict]):
    from .config import InfinoConfig, InfinoIndexConfig

    run(
        db=DBTYPE,
        db_config=InfinoConfig(
            data_path=parameters["data_path"],
            table_name=parameters["table_name"],
        ),
        db_case_config=InfinoIndexConfig(
            n_cent=parameters["n_cent"],
            nprobe=parameters["nprobe"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(InfinoFTSTypedDict)
def InfinoFTS(**parameters: Unpack[InfinoFTSTypedDict]):
    from .config import InfinoConfig, InfinoFTSConfig

    run(
        db=DBTYPE,
        db_config=InfinoConfig(
            data_path=parameters["data_path"],
            table_name=parameters["table_name"],
        ),
        db_case_config=InfinoFTSConfig(),
        **parameters,
    )
