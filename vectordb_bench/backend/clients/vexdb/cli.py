import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

# from vectordb_bench.backend.clients.api import MetricType

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    IVFFlatTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class VexDBTypedDict(CommonTypedDict):
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
            required=False,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
    maintenance_work_mem: Annotated[
        str | None,
        click.option(
            "--maintenance-work-mem",
            type=str,
            help="Sets the maximum memory to be used for maintenance operations (index creation). "
            "Can be entered as string with unit like '64GB' or as an integer number of KB."
            "This will set the parameters: max_parallel_maintenance_workers,"
            " max_parallel_workers & table(parallel_workers)",
            required=False,
        ),
    ]
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel processes per maintenance operation (index creation)",
            required=False,
        ),
    ]
    table_name: Annotated[
        str,
        click.option(
            "--table-name",
            type=str,
            help="Table name",
            default="vdbbench_table_test",
            show_default=True,
            required=False,
        ),
    ]
    partitions: Annotated[
        int | None,
        click.option(
            "--partitions",
            type=int,
            help="Set whether to use hash partitioning. A value of 0 disables partitioning, while a value greater than 0 specifies the number of partitions to use.",
            required=False,
            default=0,
            show_default=True,
        ),
    ]
    create_index_before_load: Annotated[
        bool | None,
        click.option(
            "--create-index-before-load",
            type=bool,
            help="Whether create index before load,Streaming case recommended to be true，default is false",
            required=False,
            default=False,
        ),
    ]


class VexDBIVFFlatTypedDict(VexDBTypedDict, IVFFlatTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(VexDBIVFFlatTypedDict)
def VexDBIVFFlat(
    **parameters: Unpack[VexDBIVFFlatTypedDict],
):
    from .config import VexDBConfig, VexDBIVFFlatConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.VexDB,
        db_config=VexDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            partitions=parameters["partitions"],
            table_name=parameters["table_name"],
        ),
        db_case_config=VexDBIVFFlatConfig(
            lists=parameters["lists"],
            probes=parameters["probes"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            create_index_before_load=parameters["create_index_before_load"],
        ),
        **parameters,
    )


class VexDBGRAPHINDEXTypedDict(VexDBTypedDict, HNSWFlavor1):
    col_name_list: Annotated[
        str | None,
        click.option(
            "--quantizer",
            type=str,
            help="Vector quantization method,selectable values ['none','pq','rabitq']",
            required=False,
            default="none",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(VexDBGRAPHINDEXTypedDict)
def VexDBGRAPHINDEX(
    **parameters: Unpack[VexDBGRAPHINDEXTypedDict],
):
    from .config import VexDBConfig, VexDBGRAPHINDEXConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.VexDB,
        db_config=VexDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            partitions=parameters["partitions"],
            table_name=parameters["table_name"],
        ),
        db_case_config=VexDBGRAPHINDEXConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            quantizer=parameters["quantizer"],
            ef_search=parameters["ef_search"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            create_index_before_load=parameters["create_index_before_load"],
        ),
        **parameters,
    )


class VexDBHybridANNTypedDict(VexDBTypedDict, HNSWFlavor1):
    col_name_list: Annotated[
        str | None,
        click.option(
            "--col-name-list",
            type=str,
            help="Which scalar fields will be created in hybridann index, for example: 'id'、'id, label'",
            required=True,
        ),
    ]
    hybrid_query_ivf_probes_factor: Annotated[
        int | None,
        click.option(
            "--hybrid-query-ivf-probes-factor", type=int, help="Set hybrid_query_ivf_probes_factor before select"
        ),
    ]
    vec_index_magnitudes: Annotated[
        str | None,
        click.option("--vec-index-magnitudes", type=str, help="The parameter vec_index_magnitudes in create index SQL"),
    ]
    graph_magnitude_threshold: Annotated[
        int | None,
        click.option(
            "--graph-magnitude-threshold", type=int, help="The parameter graph_magnitude_threshold in create index SQL"
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(VexDBHybridANNTypedDict)
def VexDBHybridANN(
    **parameters: Unpack[VexDBHybridANNTypedDict],
):
    from .config import VexDBConfig, VexDBHybridANNConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.VexDB,
        db_config=VexDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            partitions=parameters["partitions"],
            table_name=parameters["table_name"],
        ),
        db_case_config=VexDBHybridANNConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            create_index_before_load=parameters["create_index_before_load"],
            col_name_list=parameters["col_name_list"],
            graph_magnitude_threshold=parameters["graph_magnitude_threshold"],
            hybrid_query_ivf_probes_factor=parameters["hybrid_query_ivf_probes_factor"],
            vec_index_magnitudes=parameters["vec_index_magnitudes"],
        ),
        **parameters,
    )
