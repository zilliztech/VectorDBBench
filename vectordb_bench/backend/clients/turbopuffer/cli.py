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
from ..api import MetricType

DEFAULT_PIN_TIMEOUT = 45 * 60

ApiKeyOption = Annotated[
    str,
    click.option("--api-key", type=str, help="TurboPuffer API key", required=True),
]
RegionOption = Annotated[
    str,
    click.option(
        "--region",
        type=str,
        help="TurboPuffer region (e.g. aws-us-east-1, gcp-us-central1)",
        required=True,
    ),
]
ApiBaseUrlOption = Annotated[
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
NamespaceOption = Annotated[
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
PinTimeoutOption = Annotated[
    int,
    click.option(
        "--pin-timeout",
        type=click.IntRange(min=1),
        default=DEFAULT_PIN_TIMEOUT,
        show_default=True,
        help="Seconds to wait for TurboPuffer namespace pinning or unpinning to complete",
    ),
]


class TurboPufferTypedDict(TypedDict):
    api_key: ApiKeyOption
    region: RegionOption
    api_base_url: ApiBaseUrlOption
    namespace: NamespaceOption
    multitenant_namespace_prefix: Annotated[
        str,
        click.option(
            "--multitenant-namespace-prefix",
            type=str,
            help="Namespace prefix for CloudMultiTenantSearchCase tenant namespaces",
            required=False,
            default="vdbbench_mt_",
            show_default=True,
        ),
    ]
    scalar_payload_label_field: Annotated[
        str,
        click.option(
            "--scalar-payload-label-field",
            type=str,
            help="TurboPuffer attribute used for scalar_label payload and label filtering",
            required=False,
            default="label",
            show_default=True,
        ),
    ]
    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice([MetricType.COSINE.value, MetricType.L2.value]),
            help="TurboPuffer distance metric type",
            required=False,
            default=MetricType.COSINE.value,
            show_default=True,
        ),
    ]
    disable_backpressure: Annotated[
        bool,
        click.option(
            "--disable-backpressure/--enable-backpressure",
            type=bool,
            default=False,
            show_default=True,
            help="Disable Turbopuffer write backpressure",
        ),
    ]
    pin_namespace: Annotated[
        bool,
        click.option(
            "--pin-namespace/--no-pin-namespace",
            default=False,
            show_default=True,
            help="Pin TurboPuffer namespace(s) before benchmark workers run",
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
    pin_timeout: PinTimeoutOption


class TurboPufferIndexTypedDict(CommonTypedDict, TurboPufferTypedDict): ...


class TurboPufferUnpinTypedDict(TypedDict):
    """Options for explicit TurboPuffer namespace pinning cleanup."""

    api_key: ApiKeyOption
    region: RegionOption
    api_base_url: ApiBaseUrlOption
    namespace: NamespaceOption
    pin_timeout: PinTimeoutOption


def target_namespaces_for_pinning(parameters: TurboPufferIndexTypedDict) -> list[str]:
    if parameters.get("case_type") != "CloudMultiTenantSearchCase":
        return [parameters["namespace"]]

    namespace_prefix = parameters["multitenant_namespace_prefix"]
    tenant_prefix = parameters["tenant_prefix"]
    tenant_id_width = parameters["tenant_id_width"]
    return [
        f"{namespace_prefix}{tenant_prefix}{tenant_id:0{tenant_id_width}d}"
        for tenant_id in range(parameters["tenant_count"])
    ]


def pin_namespaces_once(parameters: TurboPufferIndexTypedDict) -> None:
    from .turbopuffer import namespace_metadata_request, wait_for_namespace_pinning

    for namespace in target_namespaces_for_pinning(parameters):
        namespace_metadata_request(
            parameters["api_key"],
            parameters["region"],
            namespace,
            "PATCH",
            {"pinning": {"replicas": parameters["pin_replicas"]}},
            parameters["api_base_url"] or None,
        )
        wait_for_namespace_pinning(
            parameters["api_key"],
            parameters["region"],
            namespace,
            parameters["pin_replicas"],
            parameters["api_base_url"] or None,
            parameters["pin_timeout"],
        )


@cli.command()
@click_parameter_decorators_from_typed_dict(TurboPufferIndexTypedDict)
def TurboPuffer(**parameters: Unpack[TurboPufferIndexTypedDict]):
    from .config import TurboPufferConfig, TurboPufferIndexConfig

    if parameters["pin_namespace"]:
        pin_namespaces_once(parameters)

    run(
        db=DB.TurboPuffer,
        db_config=TurboPufferConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]),
            region=parameters["region"],
            api_base_url=parameters["api_base_url"] or None,
            namespace=parameters["namespace"],
            multitenant_namespace_prefix=parameters["multitenant_namespace_prefix"],
            scalar_payload_label_field=parameters["scalar_payload_label_field"],
            pin_namespace=False,
            pin_replicas=parameters["pin_replicas"],
            pin_timeout=parameters["pin_timeout"],
        ),
        db_case_config=TurboPufferIndexConfig(
            metric_type=MetricType(parameters["metric_type"]),
            disable_backpressure=parameters["disable_backpressure"],
        ),
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
