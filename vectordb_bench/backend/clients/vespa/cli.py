from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class VespaTypedDict(CommonTypedDict, HNSWFlavor1):
    uri: Annotated[
        str,
        click.option("--uri", "-u", type=str, help="uri connection string", default="http://127.0.0.1"),
    ]
    port: Annotated[
        int,
        click.option("--port", "-p", type=int, help="connection port", default=8080),
    ]
    quantization: Annotated[
        str, click.option("--quantization", type=click.Choice(["none", "binary"], case_sensitive=False), default="none")
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(VespaTypedDict)
def Vespa(**params: Unpack[VespaTypedDict]):
    from .config import VespaConfig, VespaHNSWConfig

    case_params = {
        "quantization_type": params["quantization"],
        "M": params["m"],
        "efConstruction": params["ef_construction"],
        "ef": params["ef_search"],
    }

    run(
        db=DB.Vespa,
        db_config=VespaConfig(url=SecretStr(params["uri"]), port=params["port"]),
        db_case_config=VespaHNSWConfig(**{k: v for k, v in case_params.items() if v}),
        **params,
    )
