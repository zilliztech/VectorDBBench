from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class WeaviateTypedDict(CommonTypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="Weaviate api key", required=False, default=""),
    ]
    url: Annotated[
        str,
        # HTTP endpoint for Weaviate (required). Do not pass the gRPC port here.
        click.option("--url", type=str, help="Weaviate HTTP url (e.g. http://localhost:8080)", required=True),
    ]
    grpc_url: Annotated[
        str,
        # Optional: allow providing a gRPC address separately; it is currently not used by the Python client.
        click.option(
            "--grpc-url",
            type=str,
            required=False,
            default="",
            help="Optional Weaviate gRPC address (e.g. localhost:50051). Not used by this runner.",
        ),
    ]
    no_auth: Annotated[
        bool,
        click.option(
            "--no-auth",
            is_flag=True,
            help="Do not use api-key, set it to true if you are using a local setup. Default is False.",
            default=False,
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, default=16, help="HNSW index parameter m."),
    ]
    ef_construct: Annotated[
        int,
        click.option("--ef-construction", type=int, default=256, help="HNSW index parameter ef_construction"),
    ]
    ef: Annotated[
        int,
        click.option("--ef", type=int, default=256, help="HNSW index parameter ef for search"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(WeaviateTypedDict)
def Weaviate(**parameters: Unpack[WeaviateTypedDict]):
    from .config import WeaviateConfig, WeaviateIndexConfig

    # Guard: ensure the HTTP URL includes a scheme to avoid requests InvalidSchema errors
    http_url = parameters["url"]
    if not (http_url.startswith("http://") or http_url.startswith("https://")):
        http_url = f"http://{http_url}"

    run(
        db=DB.WeaviateCloud,
        db_config=WeaviateConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]) if parameters["api_key"] != "" else SecretStr("-"),
            url=SecretStr(http_url),
            no_auth=parameters["no_auth"],
            grpc_url=SecretStr(parameters["grpc_url"]) if parameters.get("grpc_url") else None,
        ),
        db_case_config=WeaviateIndexConfig(
            efConstruction=parameters["ef_construction"],
            maxConnections=parameters["m"],
            ef=parameters["ef"],
        ),
        **parameters,
    )
