from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run


class GarnetTypedDict(CommonTypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Garnet host", default="127.0.0.1")]
    port: Annotated[int, click.option("--port", type=int, help="Garnet port", default=6379)]
    username: Annotated[str | None, click.option("--username", type=str, help="Garnet username")]
    password: Annotated[str | None, click.option("--password", type=str, help="Garnet password")]

    max_degree: Annotated[int, click.option("--max-degree", type=int, help="Maximum graph degree", required=True)]
    l_build: Annotated[int, click.option("--l-build", type=int, help="Build list size", default=128)]
    l_search: Annotated[int, click.option("--l-search", type=int, help="Search list size", default=15)]
    filter_scale: Annotated[
        int, click.option("--filter-scale", type=int, help="Adaptive filter scale factor", default=16)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(GarnetTypedDict)
def Garnet(**parameters: Unpack[GarnetTypedDict]):
    from .config import GarnetDBCaseConfig, GarnetDBConfig

    run(
        db=DB.Garnet,
        db_config=GarnetDBConfig(
            db_label=parameters["db_label"],
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
            username=SecretStr(parameters["username"]) if parameters["username"] else None,
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
        ),
        db_case_config=GarnetDBCaseConfig(
            max_degree=parameters["max_degree"],
            l_build=parameters["l_build"],
            l_search=parameters["l_search"],
            filter_scale=parameters["filter_scale"],
        ),
        **parameters,
    )
