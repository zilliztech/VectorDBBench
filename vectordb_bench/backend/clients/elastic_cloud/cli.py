from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.ElasticCloud


class ElasticCloudTypedDict(TypedDict):
    cloud_id: Annotated[
        str,
        click.option("--cloud-id", type=str, help="Elastic Cloud ID", required=True),
    ]
    password: Annotated[
        str,
        click.option("--password", type=str, help="Elastic Cloud password", required=True),
    ]
    number_of_shards: Annotated[
        int,
        click.option(
            "--number-of-shards",
            type=int,
            help="Number of shards",
            required=False,
            default=1,
            show_default=True,
        ),
    ]
    number_of_replicas: Annotated[
        int,
        click.option(
            "--number-of-replicas",
            type=int,
            help="Number of replicas",
            required=False,
            default=0,
            show_default=True,
        ),
    ]
    refresh_interval: Annotated[
        str,
        click.option(
            "--refresh-interval",
            type=str,
            help="Index refresh interval",
            required=False,
            default="30s",
            show_default=True,
        ),
    ]
    merge_max_thread_count: Annotated[
        int,
        click.option(
            "--merge-max-thread-count",
            type=int,
            help="Maximum thread count for merge",
            required=False,
            default=8,
            show_default=True,
        ),
    ]
    use_force_merge: Annotated[
        bool,
        click.option(
            "--use-force-merge",
            type=bool,
            help="Whether to use force merge",
            required=False,
            default=True,
            show_default=True,
        ),
    ]
    use_routing: Annotated[
        bool,
        click.option(
            "--use-routing",
            type=bool,
            help="Whether to use routing",
            required=False,
            default=False,
            show_default=True,
        ),
    ]
    use_rescore: Annotated[
        bool,
        click.option(
            "--use-rescore",
            type=bool,
            help="Whether to use rescore",
            required=False,
            default=False,
            show_default=True,
        ),
    ]
    oversample_ratio: Annotated[
        float,
        click.option(
            "--oversample-ratio",
            type=float,
            help="Oversample ratio for rescore",
            required=False,
            default=2.0,
            show_default=True,
        ),
    ]


class ElasticCloudHNSWTypedDict(CommonTypedDict, ElasticCloudTypedDict):
    m: Annotated[
        int,
        click.option(
            "--m",
            type=int,
            help="HNSW M parameter",
            required=False,
            default=16,
            show_default=True,
        ),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="HNSW efConstruction parameter",
            required=False,
            default=100,
            show_default=True,
        ),
    ]
    num_candidates: Annotated[
        int,
        click.option(
            "--num-candidates",
            type=int,
            help="Number of candidates for search",
            required=False,
            default=100,
            show_default=True,
        ),
    ]
    element_type: Annotated[
        str,
        click.option(
            "--element-type",
            type=click.Choice(["float", "byte"], case_sensitive=False),
            help="Element type for vectors (float: 4 bytes, byte: 1 byte)",
            required=False,
            default="float",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(ElasticCloudHNSWTypedDict)
def ElasticCloudHNSW(**parameters: Unpack[ElasticCloudHNSWTypedDict]):
    from ..api import IndexType
    from .config import ElasticCloudConfig, ElasticCloudIndexConfig, ESElementType

    run(
        db=DBTYPE,
        db_config=ElasticCloudConfig(
            db_label=parameters["db_label"],
            cloud_id=SecretStr(parameters["cloud_id"]),
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=ElasticCloudIndexConfig(
            index=IndexType.ES_HNSW,
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            num_candidates=parameters["num_candidates"],
            element_type=ESElementType(parameters["element_type"]),
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            refresh_interval=parameters["refresh_interval"],
            merge_max_thread_count=parameters["merge_max_thread_count"],
            use_force_merge=parameters["use_force_merge"],
            use_routing=parameters["use_routing"],
            use_rescore=parameters["use_rescore"],
            oversample_ratio=parameters["oversample_ratio"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(ElasticCloudHNSWTypedDict)
def ElasticCloudHNSWInt8(**parameters: Unpack[ElasticCloudHNSWTypedDict]):
    from ..api import IndexType
    from .config import ElasticCloudConfig, ElasticCloudIndexConfig, ESElementType

    run(
        db=DBTYPE,
        db_config=ElasticCloudConfig(
            db_label=parameters["db_label"],
            cloud_id=SecretStr(parameters["cloud_id"]),
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=ElasticCloudIndexConfig(
            index=IndexType.ES_HNSW_INT8,
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            num_candidates=parameters["num_candidates"],
            element_type=ESElementType(parameters["element_type"]),
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            refresh_interval=parameters["refresh_interval"],
            merge_max_thread_count=parameters["merge_max_thread_count"],
            use_force_merge=parameters["use_force_merge"],
            use_routing=parameters["use_routing"],
            use_rescore=parameters["use_rescore"],
            oversample_ratio=parameters["oversample_ratio"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(ElasticCloudHNSWTypedDict)
def ElasticCloudHNSWInt4(**parameters: Unpack[ElasticCloudHNSWTypedDict]):
    from ..api import IndexType
    from .config import ElasticCloudConfig, ElasticCloudIndexConfig, ESElementType

    run(
        db=DBTYPE,
        db_config=ElasticCloudConfig(
            db_label=parameters["db_label"],
            cloud_id=SecretStr(parameters["cloud_id"]),
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=ElasticCloudIndexConfig(
            index=IndexType.ES_HNSW_INT4,
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            num_candidates=parameters["num_candidates"],
            element_type=ESElementType(parameters["element_type"]),
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            refresh_interval=parameters["refresh_interval"],
            merge_max_thread_count=parameters["merge_max_thread_count"],
            use_force_merge=parameters["use_force_merge"],
            use_routing=parameters["use_routing"],
            use_rescore=parameters["use_rescore"],
            oversample_ratio=parameters["oversample_ratio"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(ElasticCloudHNSWTypedDict)
def ElasticCloudHNSWBBQ(**parameters: Unpack[ElasticCloudHNSWTypedDict]):
    from ..api import IndexType
    from .config import ElasticCloudConfig, ElasticCloudIndexConfig, ESElementType

    run(
        db=DBTYPE,
        db_config=ElasticCloudConfig(
            db_label=parameters["db_label"],
            cloud_id=SecretStr(parameters["cloud_id"]),
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=ElasticCloudIndexConfig(
            index=IndexType.ES_HNSW_BBQ,
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            num_candidates=parameters["num_candidates"],
            element_type=ESElementType(parameters["element_type"]),
            number_of_shards=parameters["number_of_shards"],
            number_of_replicas=parameters["number_of_replicas"],
            refresh_interval=parameters["refresh_interval"],
            merge_max_thread_count=parameters["merge_max_thread_count"],
            use_force_merge=parameters["use_force_merge"],
            use_routing=parameters["use_routing"],
            use_rescore=parameters["use_rescore"],
            oversample_ratio=parameters["oversample_ratio"],
        ),
        **parameters,
    )
