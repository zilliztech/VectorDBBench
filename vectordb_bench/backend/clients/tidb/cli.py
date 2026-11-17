from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run
from .config import TiDBConfig, TiDBIndexConfig, TiDBSPFreshIndexConfig


class TiDBTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            default="root",
            show_default=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            default="",
            show_default=True,
            help="Password",
        ),
    ]
    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            default="127.0.0.1",
            show_default=True,
            help="Db host",
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=4000,
            show_default=True,
            help="Db Port",
        ),
    ]
    db_name: Annotated[
        str,
        click.option(
            "--db-name",
            type=str,
            default="test",
            show_default=True,
            help="Db name",
        ),
    ]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            default=False,
            show_default=True,
            is_flag=True,
            help="Enable or disable SSL, for TiDB Serverless SSL must be enabled",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TiDBTypedDict)
def TiDB(
    **parameters: Unpack[TiDBTypedDict],
):
    _run_tidb(DB.TiDB, TiDBIndexConfig(), parameters)


class TiDBSPFreshTypedDict(TiDBTypedDict):
    spfresh_build_mode: Annotated[
        str,
        click.option(
            "--spfresh-build-mode",
            type=click.Choice(["inline", "non-inline"]),
            default="inline",
            show_default=True,
            help="Create the SPFresh index with the table or after loading the data",
        ),
    ]
    spfresh_vector_index_param: Annotated[
        str | None,
        click.option(
            "--spfresh-vector-index-param",
            type=str,
            default=None,
            help="Raw VECTOR_INDEX_PARAM for index creation; does not update an index with --skip-load",
        ),
    ]
    spfresh_wait_timeout: Annotated[
        float,
        click.option(
            "--spfresh-wait-timeout",
            type=click.FloatRange(min=0, min_open=True),
            default=600.0,
            show_default=True,
            help="Maximum seconds to wait for the SPFresh checkpoint after loading",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TiDBSPFreshTypedDict)
def TiDBSPFresh(**parameters: Unpack[TiDBSPFreshTypedDict]):
    _run_tidb(
        DB.TiDBSPFresh,
        TiDBSPFreshIndexConfig(
            spfresh_build_mode=parameters["spfresh_build_mode"],
            spfresh_vector_index_param=parameters["spfresh_vector_index_param"],
            spfresh_wait_timeout=parameters["spfresh_wait_timeout"],
        ),
        parameters,
    )


def _run_tidb(db: DB, db_case_config: TiDBIndexConfig, parameters: dict):
    run(
        db=db,
        db_config=TiDBConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            ssl=parameters["ssl"],
        ),
        db_case_config=db_case_config,
        **parameters,
    )
