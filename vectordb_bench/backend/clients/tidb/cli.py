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
    min_partition_size: Annotated[
        int | None,
        click.option(
            "--min-partition-size",
            type=click.IntRange(min=1),
            default=None,
            help="Minimum partition size at index creation; omitted uses the server default",
        ),
    ]
    max_partition_size: Annotated[
        int | None,
        click.option(
            "--max-partition-size",
            type=click.IntRange(min=2),
            default=None,
            help="Maximum partition size at index creation; omitted uses the server default",
        ),
    ]
    write_beam_size: Annotated[
        int | None,
        click.option(
            "--write-beam-size",
            type=click.IntRange(1, 4096),
            default=None,
            help="Write routing beam size at index creation; omitted uses the server default",
        ),
    ]
    search_beam_size: Annotated[
        int | None,
        click.option(
            "--search-beam-size",
            type=click.IntRange(1, 4096),
            default=None,
            help="Search beam size; also updates the existing index with --skip-load",
        ),
    ]
    oversample_factor: Annotated[
        float | None,
        click.option(
            "--oversample-factor",
            type=click.FloatRange(1, 16),
            default=None,
            help="Rerank candidate oversampling factor; also applied with --skip-load",
        ),
    ]
    hamming_factor: Annotated[
        int | None,
        click.option(
            "--hamming-factor",
            type=click.IntRange(1, 256),
            default=None,
            help="Hamming candidate factor, >= rabitq4-factor; also applied with --skip-load",
        ),
    ]
    rabitq4_factor: Annotated[
        int | None,
        click.option(
            "--rabitq4-factor",
            type=click.IntRange(1, 256),
            default=None,
            help="Four-bit candidate factor, <= hamming-factor; also applied with --skip-load",
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
            spfresh_wait_timeout=parameters["spfresh_wait_timeout"],
            min_partition_size=parameters["min_partition_size"],
            max_partition_size=parameters["max_partition_size"],
            write_beam_size=parameters["write_beam_size"],
            search_beam_size=parameters["search_beam_size"],
            oversample_factor=parameters["oversample_factor"],
            hamming_factor=parameters["hamming_factor"],
            rabitq4_factor=parameters["rabitq4_factor"],
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
