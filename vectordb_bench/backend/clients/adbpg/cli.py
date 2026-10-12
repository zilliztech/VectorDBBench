import os
from typing import Annotated, Any, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .config import parse_adbpg_parameter_group


class AdbpgParameterGroupType(click.ParamType):
    name = "YAML_MAPPING"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> dict[str, Any]:
        try:
            return dict(parse_adbpg_parameter_group(value, param.name if param is not None else self.name))
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


ADBPG_PARAMETER_GROUP = AdbpgParameterGroupType()


class AdbpgTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Postgres database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
    build_parameters: Annotated[
        dict[str, Any],
        click.option(
            "--build-parameters",
            type=ADBPG_PARAMETER_GROUP,
            default="{}",
            help="Build reloption/GUC mapping as YAML or JSON",
            show_default=True,
        ),
    ]
    search_parameters: Annotated[
        dict[str, Any],
        click.option(
            "--search-parameters",
            type=ADBPG_PARAMETER_GROUP,
            default="{}",
            help="Search reloption/GUC mapping as YAML or JSON",
            show_default=True,
        ),
    ]
    autotune_parameters: Annotated[
        dict[str, Any] | None,
        click.option(
            "--autotune-parameters",
            type=ADBPG_PARAMETER_GROUP,
            help="Autotune mapping as YAML or JSON; omit to disable",
            default=None,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AdbpgTypedDict)
def AdbpgNova(**parameters: Unpack[AdbpgTypedDict]):
    from .config import AdbpgConfig, AdbpgIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.Adbpg,
        db_config=AdbpgConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=AdbpgIndexConfig(
            build_parameters=parameters["build_parameters"],
            search_parameters=parameters["search_parameters"],
            autotune_parameters=parameters["autotune_parameters"],
        ),
        **parameters,
    )
