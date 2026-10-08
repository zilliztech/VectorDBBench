from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

from .config import _DEFAULT_CACHE_BUDGET_BYTES

DBTYPE = DB.Infino


def _parse_kv_list(_ctx, _param, values) -> dict[str, str]:  # noqa: ANN001
    """Parse repeatable or comma-separated key=value items into a dict."""
    parsed: dict[str, str] = {}
    for item in values or ():
        for part in (p.strip() for p in str(item).split(",")):
            if not part:
                continue
            if "=" not in part:
                msg = f"Expect key=value, got: {part}"
                raise click.BadParameter(msg)
            k, v = part.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


# Connection options shared across the command's parameter set.
_data_path_option = click.option(
    "--data-path", type=str, default="/tmp/vectordb_bench/infino", help="Infino catalog directory"
)
_cache_budget_option = click.option(
    "--cache-budget-bytes",
    type=int,
    default=_DEFAULT_CACHE_BUDGET_BYTES,
    help="Disk-cache ceiling in bytes; raise for corpora larger than the cache",
)
_cache_dir_option = click.option("--cache-dir", type=str, default=None, help="Infino disk-cache directory")
_storage_option_option = click.option(
    "--storage-option",
    "storage_options",
    type=str,
    multiple=True,
    callback=_parse_kv_list,
    help="Object-store option as key=value (repeatable or comma-separated), e.g. region=us-east-1",
)


class InfinoCommonTypedDict(CommonTypedDict):
    data_path: Annotated[str, _data_path_option]
    cache_budget_bytes: Annotated[int, _cache_budget_option]
    cache_dir: Annotated[str, _cache_dir_option]
    storage_options: Annotated[dict, _storage_option_option]


class InfinoTypedDict(InfinoCommonTypedDict):
    table_name: Annotated[
        str,
        click.option("--table-name", type=str, default="vdbbench_infino", help="Infino table name"),
    ]
    search_mode: Annotated[
        str,
        click.option(
            "--search-mode",
            type=click.Choice(["ivf", "hnsw_ivf"]),
            default="hnsw_ivf",
            help="Vector serving path, bridged to the engine config: "
            "hnsw_ivf (default; resident HNSW graph with ivf fallback) or ivf",
        ),
    ]
    ef: Annotated[
        int,
        click.option(
            "--ef",
            type=int,
            default=0,
            help="Serve-time HNSW beam (search_mode=hnsw_ivf), bridged to "
            "vector.hnsw_ef_search. 0 (default) uses the graph's stamped k->ef "
            "curve; a positive value fixes the beam. Sweep it across runs "
            "(one value per run) to trace the recall/latency curve without a rebuild.",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(InfinoTypedDict)
def Infino(**parameters: Unpack[InfinoTypedDict]):
    from .config import InfinoConfig, InfinoIndexConfig

    run(
        db=DBTYPE,
        db_config=InfinoConfig(
            db_label=parameters["db_label"],
            data_path=parameters["data_path"],
            table_name=parameters["table_name"],
            cache_budget_bytes=parameters["cache_budget_bytes"],
            cache_dir=parameters["cache_dir"],
            storage_options=parameters["storage_options"] or None,
        ),
        db_case_config=InfinoIndexConfig(search_mode=parameters["search_mode"], ef=parameters["ef"]),
        **parameters,
    )
