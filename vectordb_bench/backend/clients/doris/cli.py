from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    HNSWBaseTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


def _parse_kv_list(_ctx, _param, values):  # noqa: ANN001
    """Parse repeatable or comma-separated key=value items into a dict.

    Accepts any of the following forms (and mixtures thereof):
      --index-prop a=1 --index-prop b=2
      --index-prop a=1,b=2
      --index-prop a=1,b=2 --index-prop c=3
    """
    parsed: dict[str, str] = {}
    if not values:
        return parsed
    for item in values:
        # allow comma-separated list in a single occurrence
        parts = [p.strip() for p in str(item).split(",") if p and p.strip()]
        for part in parts:
            if "=" not in part:
                msg = f"Expect key=value, got: {part}"
                raise click.BadParameter(msg)
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                msg = f"Empty key in: {part}"
                raise click.BadParameter(msg)
            parsed[k] = v
    return parsed


class DorisTypedDict(CommonTypedDict, HNSWBaseTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            default="root",
            show_default=True,
            required=True,
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
            required=True,
            help="Db host",
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=9030,
            show_default=True,
            required=True,
            help="Query Port",
        ),
    ]
    http_port: Annotated[
        int,
        click.option(
            "--http-port",
            type=int,
            default=8030,
            show_default=True,
            required=True,
            help="Http Port",
        ),
    ]
    db_name: Annotated[
        str,
        click.option(
            "--db-name",
            type=str,
            default="test",
            show_default=True,
            required=True,
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
            help="Enable or disable SSL, for Doris Serverless SSL must be enabled",
        ),
    ]
    index_prop: Annotated[
        dict,
        click.option(
            "--index-prop",
            type=str,
            multiple=True,
            help="Extra index PROPERTY as key=value (repeatable or comma-separated, e.g. a=1,b=2)",
            callback=_parse_kv_list,
        ),
    ]
    session_var: Annotated[
        dict,
        click.option(
            "--session-var",
            type=str,
            multiple=True,
            help="Session variable key=value applied to each SQL session (repeatable or comma-separated)",
            callback=_parse_kv_list,
        ),
    ]
    stream_load_rows_per_batch: Annotated[
        int | None,
        click.option(
            "--stream-load-rows-per-batch",
            type=int,
            required=False,
            help="Rows per single stream load request; default uses NUM_PER_BATCH",
        ),
    ]
    no_index: Annotated[
        bool,
        click.option(
            "--no-index",
            is_flag=True,
            default=False,
            show_default=True,
            help="Create table without ANN index",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(DorisTypedDict)
def Doris(
    **parameters: Unpack[DorisTypedDict],
):
    from .config import DorisCaseConfig, DorisConfig

    # Merge explicit HNSW params into index properties using Doris naming
    index_properties: dict[str, str] = {}
    index_properties.update(parameters.get("index_prop", {}) or {})
    if parameters.get("m") is not None:
        index_properties.setdefault("max_degree", str(parameters["m"]))
    if parameters.get("ef_construction") is not None:
        index_properties.setdefault("ef_construction", str(parameters["ef_construction"]))

    session_vars: dict[str, str] = parameters.get("session_var", {}) or {}

    run(
        db=DB.Doris,
        db_config=DorisConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            http_port=parameters["http_port"],
            db_name=parameters["db_name"],
            ssl=parameters["ssl"],
        ),
        # metric_type should come from the dataset; Assembler will set it on the case config.
        db_case_config=DorisCaseConfig(
            index_properties=index_properties,
            session_vars=session_vars,
            stream_load_rows_per_batch=parameters.get("stream_load_rows_per_batch"),
            no_index=parameters.get("no_index", False),
        ),
        **parameters,
    )
