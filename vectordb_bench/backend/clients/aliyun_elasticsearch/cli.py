from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run

from .config import AliyunElasticsearchConfig, AliyunElasticsearchIndexConfig, AliyunESQueryWireFormat


class AliyunElasticsearchTypedDict(CommonTypedDict):
    scheme: Annotated[
        str,
        click.option("--scheme", type=click.Choice(["http", "https"]), default="http", show_default=True),
    ]
    host: Annotated[str, click.option("--host", type=str, required=True, help="Elasticsearch hostname.")]
    port: Annotated[int, click.option("--port", type=click.IntRange(1, 65535), default=9200, show_default=True)]
    user: Annotated[str, click.option("--user", type=str, default="elastic", show_default=True)]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            required=True,
            envvar="VDBBENCH_ES_PASSWORD",
            help="Elasticsearch password.",
        ),
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", type=str, default="vdb_bench_indice", show_default=True),
    ]
    query_wire_format: Annotated[
        str,
        click.option(
            "--query-wire-format",
            type=click.Choice([value.value for value in AliyunESQueryWireFormat]),
            default=AliyunESQueryWireFormat.base64_f32be.value,
            show_default=True,
            help="Query encoding; wire-format environment overrides take precedence.",
        ),
    ]
    m: Annotated[int, click.option("--m", type=click.IntRange(min=1), default=16, show_default=True)]
    ef_construction: Annotated[
        int,
        click.option("--ef-construction", type=click.IntRange(min=1), default=100, show_default=True),
    ]
    num_candidates: Annotated[
        int,
        click.option("--num-candidates", type=click.IntRange(min=1), default=100, show_default=True),
    ]
    use_rescore: Annotated[
        bool,
        click.option("--use-rescore/--no-use-rescore", default=False, show_default=True),
    ]
    oversample_ratio: Annotated[
        float,
        click.option("--oversample-ratio", type=click.FloatRange(min=1), default=2.0, show_default=True),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AliyunElasticsearchTypedDict)
def AliyunElasticsearch(**parameters: Unpack[AliyunElasticsearchTypedDict]):
    """Benchmark Aliyun Elasticsearch, including existing native_hnsw indexes."""
    run(
        db=DB.AliyunElasticsearch,
        db_config=AliyunElasticsearchConfig(
            db_label=parameters["db_label"],
            scheme=parameters["scheme"],
            host=parameters["host"],
            port=parameters["port"],
            user=parameters["user"],
            password=SecretStr(parameters["password"]),
            index_name=parameters["index_name"],
        ),
        db_case_config=AliyunElasticsearchIndexConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            num_candidates=parameters["num_candidates"],
            use_rescore=parameters["use_rescore"],
            oversample_ratio=parameters["oversample_ratio"],
            query_wire_format=AliyunESQueryWireFormat(parameters["query_wire_format"]),
        ),
        **parameters,
    )
