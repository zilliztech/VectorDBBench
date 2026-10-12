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


class HologresHGraphTypedDict(CommonTypedDict, HologresTypedDict, HNSWFlavor5):
    quantization_method: Annotated[
        str,
        click.option(
            "--quantization-method",
            type=click.Choice(["rabitq", "sq8_uniform", "fp32"], case_sensitive=True),
            default="rabitq",
            show_default=True,
            help="Base quantization type for the HGraph index. Ignored when --no-use-reorder (fp32 is forced).",
        ),
    ]
    full_compact_max_file_size_mb: Annotated[
        int,
        click.option(
            "--full-compact-max-file-size-mb",
            type=int,
            default=16384,
            show_default=True,
            help="Max file size (MB) for full compaction of the HGraph index",
        ),
    ]
    precise_io_type: Annotated[
        str,
        click.option(
            "--precise-io-type",
            type=click.Choice(["block_memory_io", "reader_io"], case_sensitive=True),
            default="block_memory_io",
            show_default=True,
            help=(
                "Storage medium for the precise index (only effective with --use-reorder). "
                "block_memory_io: all in memory; reader_io: precise index on disk."
            ),
        ),
    ]
    use_extra_column_id: Annotated[
        bool,
        click.option(
            "--use-extra-column-id/--no-use-extra-column-id",
            is_flag=True,
            type=bool,
            default=True,
            show_default=True,
            help="Embed the primary key 'id' in the index via extra_columns to skip base-table lookups",
        ),
    ]


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
            quantization_method=parameters["quantization_method"],
            precise_io_type=parameters["precise_io_type"],
            full_compact_max_file_size_mb=parameters["full_compact_max_file_size_mb"],
            use_extra_column_id=parameters["use_extra_column_id"],
        ),
        **parameters,
    )
