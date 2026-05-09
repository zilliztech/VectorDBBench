from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB

DEFAULT_PIN_TIMEOUT = 45 * 60


class TurboPufferTypedDict(TypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="TurboPuffer API key", required=True),
    ]
    region: Annotated[
        str,
        click.option(
            "--region",
            type=str,
            help="TurboPuffer region (e.g. aws-us-east-1, gcp-us-central1)",
            required=True,
        ),
    ]
    api_base_url: Annotated[
        str,
        click.option(
            "--api-base-url",
            type=str,
            help="Override the region-based API URL",
            required=False,
            default="",
            show_default=False,
        ),
    ]
    namespace: Annotated[
        str,
        click.option(
            "--namespace",
            type=str,
            help="TurboPuffer namespace",
            required=False,
            default="vdbbench_test",
            show_default=True,
        ),
    ]
    pin_namespace: Annotated[
        bool,
        click.option(
            "--pin-namespace/--no-pin-namespace",
            default=False,
            show_default=True,
            help="Enable TurboPuffer namespace pinning before the benchmark runs",
        ),
    ]
    pin_replicas: Annotated[
        int,
        click.option(
            "--pin-replicas",
            type=click.IntRange(min=1),
            default=1,
            show_default=True,
            help="Number of TurboPuffer pinning replicas to request",
        ),
    ]
    pin_timeout: Annotated[
        int,
        click.option(
            "--pin-timeout",
            type=click.IntRange(min=1),
            default=DEFAULT_PIN_TIMEOUT,
            show_default=True,
            help="Seconds to wait for TurboPuffer namespace pinning or unpinning to complete",
        ),
    ]


class TurboPufferIndexTypedDict(CommonTypedDict, TurboPufferTypedDict): ...


class TurboPufferUnpinTypedDict(TypedDict):
    api_key: TurboPufferTypedDict.__annotations__["api_key"]
    region: TurboPufferTypedDict.__annotations__["region"]
    api_base_url: TurboPufferTypedDict.__annotations__["api_base_url"]
    namespace: TurboPufferTypedDict.__annotations__["namespace"]
    pin_timeout: TurboPufferTypedDict.__annotations__["pin_timeout"]


def pin_namespace_once(parameters: TurboPufferIndexTypedDict):
    from .turbopuffer import namespace_metadata_request, wait_for_namespace_pinning

    namespace_metadata_request(
        parameters["api_key"],
        parameters["region"],
        parameters["namespace"],
        "PATCH",
        {"pinning": {"replicas": parameters["pin_replicas"]}},
        parameters["api_base_url"] or None,
    )
    wait_for_namespace_pinning(
        parameters["api_key"],
        parameters["region"],
        parameters["namespace"],
        parameters["pin_replicas"],
        parameters["api_base_url"] or None,
        parameters["pin_timeout"],
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TurboPufferIndexTypedDict)
def TurboPuffer(**parameters: Unpack[TurboPufferIndexTypedDict]):
    from .config import TurboPufferConfig, TurboPufferIndexConfig

    if parameters["pin_namespace"]:
        pin_namespace_once(parameters)

    run(
        db=DB.TurboPuffer,
        db_config=TurboPufferConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]),
            region=parameters["region"],
            api_base_url=parameters["api_base_url"] or None,
            namespace=parameters["namespace"],
            pin_namespace=False,
            pin_replicas=parameters["pin_replicas"],
        ),
        db_case_config=TurboPufferIndexConfig(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TurboPufferUnpinTypedDict)
def TurboPufferUnpin(**parameters: Unpack[TurboPufferUnpinTypedDict]):
    from .turbopuffer import namespace_metadata_request, wait_for_namespace_pinning

    namespace_metadata_request(
        parameters["api_key"],
        parameters["region"],
        parameters["namespace"],
        "PATCH",
        {"pinning": None},
        parameters["api_base_url"] or None,
    )
    meta = wait_for_namespace_pinning(
        parameters["api_key"],
        parameters["region"],
        parameters["namespace"],
        None,
        parameters["api_base_url"] or None,
        parameters["pin_timeout"],
    )
    click.echo(f"TurboPuffer namespace unpinned: {parameters['namespace']} pinning={meta.get('pinning')}")
