from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor5,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class HologresTypedDict(CommonTypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Hologres host", required=True)]
    user: Annotated[str, click.option("--user", type=str, help="Hologres username", required=True)]
    password: Annotated[str, click.option("--password", type=str, help="Hologres password", required=True)]
    database: Annotated[str, click.option("--database", type=str, help="Hologres database name", required=True)]
    port: Annotated[int, click.option("--port", type=int, help="Hologres port", required=True)]


class HologresHGraphTypedDict(CommonTypedDict, HologresTypedDict, HNSWFlavor5): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(HologresHGraphTypedDict)
def HologresHGraph(**parameters: Unpack[HologresHGraphTypedDict]):
    from .config import HologresConfig, HologresIndexConfig

    run(
        db=DB.Hologres,
        db_config=HologresConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["database"],
        ),
        db_case_config=HologresIndexConfig(
            index=parameters["index_type"],
            max_degree=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            use_reorder=parameters["use_reorder"],
        ),
        **parameters,
    )
